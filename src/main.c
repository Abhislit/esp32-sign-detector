/*
 * main.c - ESP32-CAM dynamic sign-gesture detector.
 *
 * Architecture: a Motion History Image encodes the gesture's temporal
 * structure directly into a 64x64 image, so one small CNN classifies the whole
 * gesture in a single invoke. See docs/ARCHITECTURE.md.
 *
 * Task layout:
 *   core 1, prio 5  - capture loop. esp_camera_fb_get() blocks on the camera,
 *                     so the frame rate is the clock and the MHI stays
 *                     correct even when inference is slower than capture.
 *   core 0          - WiFi stack and HTTP server.
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

#include "esp_chip_info.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_spiram.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "nvs_flash.h"
#include "esp_camera.h"
#include "img_converters.h"
#include "esp_http_server.h"

#include "app_config.h"
#include "mhi.h"
#include "infer.h"
#include "vote.h"

static const char *TAG = "sign";

/* ======================================================================== */
/* Shared state                                                              */
/* ======================================================================== */

typedef enum {
    REC_IDLE = 0,
    REC_ARMED,     /* waiting for a motion trigger */
    REC_CAPTURING, /* collecting frames */
    REC_DONE,
} rec_state_t;

static struct {
    SemaphoreHandle_t lock;

    uint8_t mhi[MHI_PIXELS];

    /* Ring of the most recent grayscale frames, for dataset capture. */
    uint8_t *rec_ring;   /* MHI_SEQ_LEN * MHI_PIXELS, PSRAM */
    uint8_t rec_head;
    uint8_t rec_filled;

    int leader_class;
    float leader_score;
    float scores[NUM_GESTURES];
    int last_class;
    char last_label[24];
    uint32_t last_infer_ms;
    uint32_t frames;
    uint32_t fps_milli;          /* frames per second * 1000 */
    uint32_t motion_energy;

    rec_state_t rec_state;
    int rec_label;
    uint8_t *rec_buf;            /* PSRAM, RECORD_SAMPLE_BYTES */
    uint8_t rec_collected;
    SemaphoreHandle_t rec_sem;
} s;

static httpd_handle_t s_http = NULL;
static bool s_model_ready = false;
static uint32_t s_fps_t0 = 0;
static uint32_t s_fps_frames = 0;

/* ======================================================================== */
/* Camera                                                                    */
/* ======================================================================== */

/* AI-Thinker ESP32-CAM pinout. */
#define PIN_PWDN 32
#define PIN_RESET -1
#define PIN_XCLK 0
#define PIN_SIOD 26
#define PIN_SIOC 27
#define PIN_Y9 35
#define PIN_Y8 34
#define PIN_Y7 39
#define PIN_Y6 36
#define PIN_Y5 21
#define PIN_Y4 19
#define PIN_Y3 18
#define PIN_Y2 5
#define PIN_VSYNC 25
#define PIN_HREF 23
#define PIN_PCLK 22

static bool camera_init(void)
{
    camera_config_t cfg = {
        .pin_pwdn = PIN_PWDN,
        .pin_reset = PIN_RESET,
        .pin_xclk = PIN_XCLK,
        .pin_sccb_sda = PIN_SIOD,
        .pin_sccb_scl = PIN_SIOC,
        .pin_d7 = PIN_Y9,
        .pin_d6 = PIN_Y8,
        .pin_d5 = PIN_Y7,
        .pin_d4 = PIN_Y6,
        .pin_d3 = PIN_Y5,
        .pin_d2 = PIN_Y4,
        .pin_d1 = PIN_Y3,
        .pin_d0 = PIN_Y2,
        .pin_vsync = PIN_VSYNC,
        .pin_href = PIN_HREF,
        .pin_pclk = PIN_PCLK,

        .xclk_freq_hz = 20000000,
        .ledc_timer = LEDC_TIMER_0,
        .ledc_channel = LEDC_CHANNEL_0,

        /*
         * QVGA 320x240 x2 in PSRAM is ~300KB, comfortable. VGA would also fit
         * with 4MB PSRAM but costs roughly 2.5x the downscale time for no
         * benefit, because the MHI is 64x64 regardless.
         */
        .frame_size = FRAMESIZE_QVGA,
        .fb_count = 2,
        .grab_mode = CAMERA_GRAB_LATEST,
    };

    const bool has_psram = (esp_spiram_get_size() > 0);

    if (has_psram) {
        cfg.fb_location = CAMERA_FB_IN_PSRAM;
    } else {
        /*
         * No PSRAM. QVGA x2 will not fit in 520KB of internal SRAM alongside
         * the WiFi stack. Drop to QQVGA 160x120 with one internal buffer.
         * Expect a much worse accuracy ceiling - see M0 in docs/.
         */
        ESP_LOGE(TAG, "no PSRAM: falling back to QQVGA x1 in internal RAM. "
                      "Accuracy will be limited. See docs/TROUBLESHOOTING.md");
        cfg.fb_location = CAMERA_FB_IN_SRAM;
        cfg.fb_count = 1;
        cfg.frame_size = FRAMESIZE_QQVGA;
    }

    const esp_err_t err = esp_camera_init(&cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_camera_init failed: 0x%x", err);
        return false;
    }

    sensor_t *sen = esp_camera_sensor_get();
    if (sen == NULL) {
        ESP_LOGE(TAG, "no sensor found");
        return false;
    }
    ESP_LOGI(TAG, "sensor PID 0x%04x", (unsigned) sen->id.PID);

    if (sen->id.PID == OV2640_PID) {
        /* Most modules are mounted upside down relative to the silk screen. */
        sen->set_vflip(sen, 1);
        sen->set_hmirror(sen, 0);
    }

    /*
     * Cap the auto gain. At high gain the sensor produces per-pixel flicker
     * that the motion mask reads as motion, so the trigger never de-arms and
     * the MHI fills with noise. The OV2640 API is camera-agnostic enough that
     * the safe move is to leave exposure to the driver and rely on
     * MHI_MOTION_THRESHOLD instead of a hard-coded AE register poke.
     */
    sen->set_quality(sen, 12);

    ESP_LOGI(TAG, "camera up: %ux%u fb_count=%d %s",
             (unsigned) sen->cur_size->width, (unsigned) sen->cur_size->height,
             cfg.fb_count, has_psram ? "PSRAM" : "internal RAM");
    return true;
}

/* ======================================================================== */
/* WiFi                                                                      */
/* ======================================================================== */

static void wifi_event_handler(void *arg, esp_event_base_t base,
                               int32_t id, void *data)
{
    (void) arg;
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        const ip_event_got_ip_t *e = (const ip_event_got_ip_t *) data;
        ESP_LOGI(TAG, "got IP " IPSTR, IP2STR(e->ip_info.ip));
    }
}

static void wifi_start(void)
{
    wifi_init_config_t wcfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&wcfg));
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                                wifi_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                                wifi_event_handler, NULL));

    if (strlen(APP_WIFI_SSID) > 0) {
        wifi_config_t sta = { 0 };
        strncpy((char *) sta.sta.ssid, APP_WIFI_SSID, sizeof(sta.sta.ssid) - 1);
        strncpy((char *) sta.sta.password, APP_WIFI_PASS,
                sizeof(sta.sta.password) - 1);
        sta.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

        ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
        ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &sta));
        ESP_ERROR_CHECK(esp_wifi_start());
        ESP_LOGI(TAG, "STA mode, joining '%s'", APP_WIFI_SSID);
    } else {
        wifi_config_t ap = { 0 };
        strncpy((char *) ap.ap.ssid, APP_AP_SSID, sizeof(ap.ap.ssid) - 1);
        strncpy((char *) ap.ap.password, APP_AP_PASS,
                sizeof(ap.ap.password) - 1);
        ap.ap.max_connection = 4;
        ap.ap.authmode = WIFI_AUTH_WPA2_PSK;

        ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
        ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &ap));
        ESP_ERROR_CHECK(esp_wifi_start());
        ESP_LOGI(TAG, "AP mode: join '%s' / '%s', browse http://192.168.4.1",
                 APP_AP_SSID, APP_AP_PASS);
    }
}

/* ======================================================================== */
/* HTTP: helpers                                                             */
/* ======================================================================== */

static void put_u16(uint8_t *p, uint16_t v)
{
    p[0] = (uint8_t) (v & 0xFFu);
    p[1] = (uint8_t) ((v >> 8) & 0xFFu);
}

static void put_u32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t) (v & 0xFFu);
    p[1] = (uint8_t) ((v >> 8) & 0xFFu);
    p[2] = (uint8_t) ((v >> 16) & 0xFFu);
    p[3] = (uint8_t) ((v >> 24) & 0xFFu);
}

/* Encode a 64x64 grayscale tile as an 8-bit BMP. Browsers render it natively
 * and it needs no image library. Rows are written bottom-up per the BMP spec;
 * at 64 bytes per row there is no padding to worry about. */
static size_t bmp_encode(const uint8_t *gray, uint8_t *out, size_t out_cap)
{
    const uint32_t offbits = 14u + 40u + 1024u;
    const uint32_t imgsize = MHI_PIXELS;
    const uint32_t total = offbits + imgsize;
    if (out_cap < total) {
        return 0;
    }
    memset(out, 0, offbits);

    out[0] = 'B';
    out[1] = 'M';
    put_u32(out + 2, total);
    put_u32(out + 10, offbits);

    /* BITMAPINFOHEADER */
    put_u32(out + 14, 40);
    put_u32(out + 18, MHI_W);
    put_u32(out + 22, MHI_H);
    put_u16(out + 26, 1);   /* planes */
    put_u16(out + 28, 8);   /* bits per pixel */
    put_u32(out + 30, 0);   /* BI_RGB */
    put_u32(out + 34, imgsize);
    put_u32(out + 38, 2835);
    put_u32(out + 42, 2835);
    put_u32(out + 46, 256);
    put_u32(out + 50, 0);

    /* 256-entry greyscale BGRA palette, index 0 = black. */
    uint8_t *pal = out + 54;
    for (int i = 0; i < 256; i++) {
        pal[i * 4 + 0] = (uint8_t) i;
        pal[i * 4 + 1] = (uint8_t) i;
        pal[i * 4 + 2] = (uint8_t) i;
        pal[i * 4 + 3] = 0;
    }

    /* Flip rows: BMP row 0 is the bottom scanline. */
    uint8_t *px = out + offbits;
    for (int row = 0; row < MHI_H; row++) {
        memcpy(px + (size_t) row * MHI_W,
               gray + (size_t) (MHI_H - 1 - row) * MHI_W, MHI_W);
    }
    return total;
}

/*
 * Return the current frame as JPEG. Uses the sensor's hardware JPEG when
 * available and falls back to software conversion otherwise, so the endpoint
 * works on PSRAM-less boards too.
 */
static esp_err_t send_frame_jpeg(httpd_req_t *r, uint8_t quality)
{
    camera_fb_t *fb = esp_camera_fb_get();
    if (fb == NULL) {
        return httpd_resp_send_err(r, HTTPD_500_INTERNAL_SERVER_ERROR,
                                   "capture failed");
    }

    esp_err_t ret = ESP_OK;
    httpd_resp_set_hdr(r, "Cache-Control", "no-store");
    httpd_resp_set_type(r, "image/jpeg");

    if (fb->format == PIXFORMAT_JPEG) {
        httpd_resp_send(r, (const char *) fb->buf, fb->len);
    } else {
        uint8_t *jpg = NULL;
        size_t jpg_len = 0;
        if (fmt2jpg(fb->buf, fb->width, fb->height, fb->format, quality,
                    &jpg, &jpg_len) == ESP_OK && jpg != NULL) {
            httpd_resp_send(r, (const char *) jpg, jpg_len);
            free(jpg);
        } else {
            ret = httpd_resp_send_err(r, HTTPD_500_INTERNAL_SERVER_ERROR,
                                      "jpeg encode failed");
        }
    }

    esp_camera_fb_return(fb);
    return ret;
}

/* ======================================================================== */
/* HTTP: routes                                                              */
/* ======================================================================== */

static const char DASHBOARD_HTML[] =
    "<!doctype html><html><head><meta charset=utf-8>"
    "<meta name=viewport content='width=device-width,initial-scale=1'>"
    "<title>Sign Detector</title><style>"
    "body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0;"
    "padding:16px;display:flex;flex-wrap:wrap;gap:14px}"
    ".card{background:#1c1c1c;padding:12px;border-radius:8px}"
    "h1{font-size:18px;margin:0 0 10px;width:100%}"
    "img{border-radius:4px;background:#000;display:block}"
    "canvas{image-rendering:pixelated;border-radius:4px;background:#000;"
    "display:block}"
    "table{border-collapse:collapse;font-size:14px}"
    "td{padding:2px 10px}td.k{color:#888}"
    "td.big{font-size:20px;font-weight:700}"
    "button{background:#2a6;color:#fff;border:0;padding:6px 12px;"
    "border-radius:4px;cursor:pointer;margin:0 6px 6px 0}"
    ".dim{color:#777;font-size:12px}"
    "</style></head><body><h1>ESP32-CAM sign detector</h1>"
    "<div class=card><div class=dim>live frame</div>"
    "<img id=f width=256 src='/api/frame.jpg'></div>"
    "<div class=card><div class=dim>motion history image</div>"
    "<canvas id=m width=64 height=64 style='width:192px;height:192px'></canvas>"
    "</div>"
    "<div class=card><div class=dim>state</div><table>"
    "<tr><td class=k>sign</td><td class=big id=lab>-</td></tr>"
    "<tr><td class=k>score</td><td id=sc>-</td></tr>"
    "<tr><td class=k>infer</td><td id=ms>-</td></tr>"
    "<tr><td class=k>fps</td><td id=fps>-</td></tr>"
    "<tr><td class=k>motion px</td><td id=mo>-</td></tr>"
    "<tr><td class=k>model</td><td id=md>-</td></tr>"
    "</table></div>"
    "<div class=card><div class=dim>scores</div><table id=sc2></table></div>"
    "<div class=card><div class=dim>collect sample</div>"
    "<div id=lb>arm a label, then perform the gesture</div>"
    "<div id=bs style='margin-top:8px'></div></div>"
    "<script>"
    "let L=[];"
    "function rc(n){document.getElementById('lb').textContent='armed - perform "
    "gesture '+n+'...';"
    "fetch('/api/record?label='+n).then(r=>r.text()).then(t=>"
    "{document.getElementById('lb').textContent=t;});}"
    "function tick(){"
    "fetch('/api/state').then(r=>r.json()).then(s=>{"
    "document.getElementById('lab').textContent=s.label;"
    "document.getElementById('sc').textContent=s.leader_score.toFixed(3);"
    "document.getElementById('ms').textContent=s.infer_ms+' ms';"
    "document.getElementById('fps').textContent=(s.fps/1000).toFixed(1);"
    "document.getElementById('mo').textContent=s.motion_energy;"
    "document.getElementById('md').textContent=s.model;"
    "let t=document.getElementById('sc2'),h='';"
    "for(let i=0;i<s.scores.length;i++){h+='<tr><td>'+(L[i]||i)+'</td><td>'+"
    "s.scores[i].toFixed(3)+'</td></tr>';}t.innerHTML=h;"
    "let im=new Image();im.onload=()=>{let c=document.getElementById('m'),"
    "x=c.getContext('2d');x.drawImage(im,0,0,64,64);};"
    "im.src='/api/mhi.bmp?t='+Date.now();});}"
    "fetch('/api/labels').then(r=>r.json()).then(v=>{L=v;let b="
    "document.getElementById('bs');b.innerHTML='';"
    "L.forEach((n,i)=>{let e=document.createElement('button');e.textContent=i+"
    " \" \" + n;e.onclick=()=>rc(i);b.appendChild(e);});});"
    "setInterval(tick,400);tick();"
    "</script></body></html>";

static esp_err_t http_get_root(httpd_req_t *r)
{
    httpd_resp_set_type(r, "text/html");
    return httpd_resp_send(r, DASHBOARD_HTML, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t http_get_state(httpd_req_t *r)
{
    char buf[768];

    xSemaphoreTake(s.lock, portMAX_DELAY);
    int off = snprintf(buf, sizeof(buf),
        "{\"label\":\"%s\",\"last_class\":%d,\"leader_class\":%d,"
        "\"leader_score\":%.4f,\"infer_ms\":%u,\"frames\":%u,\"fps\":%u,"
        "\"motion_energy\":%u,\"model\":\"%s\",\"scores\":[",
        s.last_label[0] ? s.last_label : "-",
        s.last_class, s.leader_class, (double) s.leader_score,
        (unsigned) s.last_infer_ms, (unsigned) s.frames,
        (unsigned) s.fps_milli, (unsigned) s.motion_energy,
        s_model_ready ? "ready" : "absent");

    for (int i = 0; i < NUM_GESTURES && off > 0 &&
                     off < (int) sizeof(buf) - 16; i++) {
        off += snprintf(buf + off, sizeof(buf) - (size_t) off, "%s%.4f",
                        i ? "," : "", (double) s.scores[i]);
    }
    if (off > 0 && off < (int) sizeof(buf) - 4) {
        off += snprintf(buf + off, sizeof(buf) - (size_t) off, "]}");
    }
    xSemaphoreGive(s.lock);

    httpd_resp_set_type(r, "application/json");
    return httpd_resp_send(r, buf, (size_t) off);
}

static esp_err_t http_get_labels(httpd_req_t *r)
{
    char buf[512];
    int off = snprintf(buf, sizeof(buf), "[");
    for (int i = 0; i < NUM_GESTURES && off > 0 &&
                     off < (int) sizeof(buf) - 32; i++) {
        const char *l = infer_label(i);
        off += snprintf(buf + off, sizeof(buf) - (size_t) off,
                        "%s\"%s\"", i ? "," : "", l);
    }
    off += snprintf(buf + off, sizeof(buf) - (size_t) off, "]");
    httpd_resp_set_type(r, "application/json");
    return httpd_resp_send(r, buf, (size_t) off);
}

static esp_err_t http_get_mhi(httpd_req_t *r)
{
    static uint8_t tile[14 + 40 + 1024 + MHI_PIXELS];

    /*
     * Snapshot and encode under the lock. `tile` is static, so two concurrent
     * requests would otherwise interleave and emit a torn image. The critical
     * section is a ~5KB copy, which is a few microseconds.
     */
    xSemaphoreTake(s.lock, portMAX_DELAY);
    const size_t n = bmp_encode(s.mhi, tile, sizeof(tile));
    xSemaphoreGive(s.lock);

    if (n == 0) {
        return httpd_resp_send_err(r, HTTPD_500_INTERNAL_SERVER_ERROR,
                                   "encode overflow");
    }
    httpd_resp_set_hdr(r, "Cache-Control", "no-store");
    httpd_resp_set_type(r, "image/bmp");
    return httpd_resp_send(r, (const char *) tile, n);
}

static esp_err_t http_get_frame(httpd_req_t *r)
{
    return send_frame_jpeg(r, 12);
}

static esp_err_t http_get_stream(httpd_req_t *r)
{
    esp_err_t res = httpd_resp_set_type(
        r, "multipart/x-mixed-replace;boundary=frame");
    if (res != ESP_OK) {
        return res;
    }

    char head[96];
    while (1) {
        camera_fb_t *fb = esp_camera_fb_get();
        if (fb == NULL) {
            vTaskDelay(pdMS_TO_TICKS(50));
            continue;
        }

        uint8_t *jpg = NULL;
        size_t jpg_len = 0;

        if (fb->format == PIXFORMAT_JPEG) {
            const size_t hlen = (size_t) snprintf(head, sizeof(head),
                "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n",
                (unsigned) fb->len);
            if (httpd_resp_send_chunk(r, head, hlen) == ESP_OK &&
                httpd_resp_send_chunk(r, (const char *) fb->buf, fb->len) == ESP_OK) {
                httpd_resp_send_chunk(r, "\r\n", 2);
            }
        } else if (fmt2jpg(fb->buf, fb->width, fb->height, fb->format, 10,
                           &jpg, &jpg_len) == ESP_OK && jpg != NULL) {
            const size_t hlen = (size_t) snprintf(head, sizeof(head),
                "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n",
                (unsigned) jpg_len);
            if (httpd_resp_send_chunk(r, head, hlen) == ESP_OK &&
                httpd_resp_send_chunk(r, (const char *) jpg, jpg_len) == ESP_OK) {
                httpd_resp_send_chunk(r, "\r\n", 2);
            }
            free(jpg);
        }

        esp_camera_fb_return(fb);
    }
    return ESP_OK;
}

/*
 * Dataset capture. Arms the recorder, blocks until the capture task has
 * produced MHI_SEQ_LEN frames after a motion trigger, then returns the raw
 * grayscale sequence. pc/collect.py wraps this.
 */
static esp_err_t http_get_record(httpd_req_t *r)
{
    if (s.rec_buf == NULL) {
        return httpd_resp_send_err(r, HTTPD_500_INTERNAL_SERVER_ERROR,
                                   "recorder not allocated (no PSRAM?)");
    }

    char query[64] = "label=0";
    if (httpd_req_get_url_query_len(r) > 0 &&
        httpd_req_get_url_query_len(r) < (int) sizeof(query)) {
        httpd_req_get_url_query_str(r, query, sizeof(query));
    }
    const char *tag = strstr(query, "label=");
    const int label = tag ? atoi(tag + 6) : 0;

    xSemaphoreTake(s.lock, portMAX_DELAY);
    s.rec_label = label;
    s.rec_collected = 0;
    s.rec_state = REC_ARMED;
    xSemaphoreGive(s.lock);

    /* Nothing is written to the socket yet, so the reply type is still
     * changeable after the wait below. */
    const int32_t got = xSemaphoreTake(s.rec_sem, pdMS_TO_TICKS(20000));

    xSemaphoreTake(s.lock, portMAX_DELAY);
    const rec_state_t st = s.rec_state;
    s.rec_state = REC_IDLE;
    xSemaphoreGive(s.lock);

    if (got != pdTRUE || st != REC_DONE) {
        return httpd_resp_send(r, "timeout: no motion detected",
                               HTTPD_RESP_USE_STRLEN);
    }

    char hdr[128];
    snprintf(hdr, sizeof(hdr),
             "label=%d bytes=%d seq=%d w=%d h=%d threshold=%d decay=%d",
             label, RECORD_SAMPLE_BYTES, MHI_SEQ_LEN, MHI_W, MHI_H,
             MHI_MOTION_THRESHOLD, MHI_DECAY_NUM);
    httpd_resp_set_hdr(r, "X-Sample-Header", hdr);
    httpd_resp_set_type(r, "application/octet-stream");
    return httpd_resp_send(r, (const char *) s.rec_buf, RECORD_SAMPLE_BYTES);
}

static esp_err_t http_get_selftest(httpd_req_t *r)
{
    infer_selftest_t st;
    const bool ran = infer_selftest(&st);

    char out[320];
    int n;
    if (!ran) {
        n = snprintf(out, sizeof(out),
            "{\"ran\":false,\"reason\":\"%s\"}",
            s_model_ready ? "no src/golden_vector.h; run pc/train.py"
                          : "no model loaded");
    } else {
        const bool pass = st.class_ok && st.score_ok;
        n = snprintf(out, sizeof(out),
            "{\"ran\":true,\"pass\":%s,"
            "\"class_match\":%s,\"score_match\":%s,"
            "\"got_class\":%d,\"got_label\":\"%s\","
            "\"want_class\":%d,\"want_label\":\"%s\","
            "\"got_score\":%.4f,\"want_score\":%.4f,\"infer_ms\":%u}",
            pass ? "true" : "false",
            st.class_ok ? "true" : "false",
            st.score_ok ? "true" : "false",
            st.got_class, infer_label(st.got_class),
            st.want_class, infer_label(st.want_class),
            (double) st.got_score, (double) st.want_score,
            (unsigned) infer_last_ms());
    }

    httpd_resp_set_type(r, "application/json");
    return httpd_resp_send(r, out, (size_t) n);
}

static esp_err_t http_get_info(httpd_req_t *r)
{
    chip_info_t ci;
    esp_chip_info(&ci);

    static const char fmt[] =
        "{\n"
        "  \"mcu_cores\": %d,\n"
        "  \"flash_mb\": %d,\n"
        "  \"psram_bytes\": %u,\n"
        "  \"heap_internal_free_total\": [%u, %u],\n"
        "  \"heap_psram_free_total\": [%u, %u],\n"
        "  \"mhi\": {\"w\": %d, \"h\": %d, \"seq_len\": %d, "
        "\"motion_threshold\": %d, \"decay_num\": %d, \"normalize\": %d},\n"
        "  \"inference\": {\"available\": %s, \"arena_bytes\": %d, "
        "\"classes\": %d, \"last_ms\": %u},\n"
        "  \"vote\": {\"window\": %d, \"min_wins\": %d, \"min_score\": %.2f, "
        "\"cooldown\": %d},\n"
        "  \"trigger_pixels\": %d\n"
        "}\n";

    char out[1024];
    const int n = snprintf(out, sizeof(out), fmt,
                           ci.cores, ci.flash_size / (1024 * 1024),
                           (unsigned) esp_spiram_get_size(),
                           (unsigned) heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                           (unsigned) heap_caps_get_total_size(MALLOC_CAP_INTERNAL),
                           (unsigned) heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
                           (unsigned) heap_caps_get_total_size(MALLOC_CAP_SPIRAM),
                           MHI_W, MHI_H, MHI_SEQ_LEN, MHI_MOTION_THRESHOLD,
                           MHI_DECAY_NUM, MHI_NORMALIZE,
                           s_model_ready ? "true" : "false", TENSOR_ARENA_SIZE,
                           NUM_GESTURES, (unsigned) infer_last_ms(),
                           VOTE_WINDOW, VOTE_MIN_WINS, (double) VOTE_MIN_SCORE,
                           VOTE_COOLDOWN, MOTION_TRIGGER_PIXELS);

    if (n <= 0 || n >= (int) sizeof(out)) {
        return httpd_resp_send_err(r, HTTPD_500_INTERNAL_SERVER_ERROR,
                                   "info overflow");
    }

    httpd_resp_set_type(r, "application/json");
    return httpd_resp_send(r, out, (size_t) n);
}

static void http_start(void)
{
    httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
    cfg.stack_size = 12288;
    cfg.server_port = 80;
    cfg.max_uri_handlers = 12;    cfg.lru_purge_enable = true;

    ESP_ERROR_CHECK(httpd_start(&s_http, &cfg));

    static const httpd_uri_t routes[] = {
        { .uri = "/",              .method = HTTP_GET, .handler = http_get_root },
        { .uri = "/api/state",     .method = HTTP_GET, .handler = http_get_state },
        { .uri = "/api/labels",    .method = HTTP_GET, .handler = http_get_labels },
        { .uri = "/api/info",      .method = HTTP_GET, .handler = http_get_info },
        { .uri = "/api/mhi.bmp",   .method = HTTP_GET, .handler = http_get_mhi },
        { .uri = "/api/frame.jpg", .method = HTTP_GET, .handler = http_get_frame },
        { .uri = "/api/stream",    .method = HTTP_GET, .handler = http_get_stream },
        { .uri = "/api/record",    .method = HTTP_GET, .handler = http_get_record },
        { .uri = "/api/selftest",  .method = HTTP_GET, .handler = http_get_selftest },
    };

    for (size_t i = 0; i < sizeof(routes) / sizeof(routes[0]); i++) {
        ESP_ERROR_CHECK(httpd_register_uri_handler(s_http, &routes[i]));
    }
    ESP_LOGI(TAG, "http server up");
}

/* ======================================================================== */
/* Capture task                                                              */
/* ======================================================================== */

#define REC_PREROLL 3

static void rec_ring_push(const uint8_t *frame)
{
    if (s.rec_ring == NULL) {
        return;
    }
    memcpy(s.rec_ring + (size_t) s.rec_head * MHI_PIXELS, frame, MHI_PIXELS);
    s.rec_head = (uint8_t) ((s.rec_head + 1u) % MHI_SEQ_LEN);
    if (s.rec_filled < MHI_SEQ_LEN) {
        s.rec_filled++;
    }
}

/* Copy the ring into `dst` oldest-first, padding the front if the ring has
 * not wrapped yet (only possible in the first fraction of a second). */
static void rec_ring_snapshot(uint8_t *dst)
{
    const uint8_t n = s.rec_filled;
    for (uint8_t i = 0; i < n; i++) {
        const uint8_t idx =
            (uint8_t) ((s.rec_head + MHI_SEQ_LEN - n + i) % MHI_SEQ_LEN);
        memcpy(dst + (size_t) i * MHI_PIXELS,
               s.rec_ring + (size_t) idx * MHI_PIXELS, MHI_PIXELS);
    }
    for (uint8_t i = n; i < MHI_SEQ_LEN; i++) {
        memcpy(dst + (size_t) i * MHI_PIXELS, dst, MHI_PIXELS);
    }
}

static void capture_task(void *arg)
{
    (void) arg;

    mhi_ctx_t mhi_ctx;
    vote_ctx_t vote_ctx;
    mhi_init(&mhi_ctx);
    vote_init(&vote_ctx);

    static uint8_t gray[MHI_PIXELS];
    static uint8_t mhi[MHI_PIXELS];
    bool primed = false;

    s_fps_t0 = (uint32_t) (esp_timer_get_time() / 1000);

    while (1) {
        camera_fb_t *fb = esp_camera_fb_get();
        if (fb == NULL) {
            ESP_LOGW(TAG, "fb_get failed");
            vTaskDelay(pdMS_TO_TICKS(20));
            continue;
        }

        /* --- preprocess: the timing-critical path ---------------------- */
        mhi_luma_downscale(fb->buf, fb->width, fb->height, gray, MHI_W, MHI_H);
        if (primed) {
            mhi_step_normalized(&mhi_ctx, gray, mhi);
        } else {
            mhi_prime(&mhi_ctx, gray);
            memset(mhi, 0, MHI_PIXELS);
            primed = true;
        }
        rec_ring_push(gray);

        /* --- inference + temporal vote --------------------------------- */
        int emitted_cls = -1;
        float emitted_score = 0.0f;

        if (s_model_ready) {
            float scores[NUM_GESTURES];
            const int n = infer_scores(mhi, scores, NUM_GESTURES);
            if (n > 0) {
                int best = 0;
                for (int i = 1; i < n; i++) {
                    if (scores[i] > scores[best]) {
                        best = i;
                    }
                }
                if (vote_push(&vote_ctx, best, scores[best],
                              &emitted_cls, &emitted_score)) {
                    ESP_LOGI(TAG, "SIGN %s (%.3f)", infer_label(emitted_cls),
                             (double) emitted_score);
                }

                xSemaphoreTake(s.lock, portMAX_DELAY);
                memcpy(s.scores, scores, sizeof(s.scores));
                s.leader_class = vote_leader(&vote_ctx);
                s.leader_score = scores[best];
                s.last_class = best;
                snprintf(s.last_label, sizeof(s.last_label), "%s",
                         infer_label(best));
                s.last_infer_ms = infer_last_ms();
                xSemaphoreGive(s.lock);
            }
        }

        /* --- publish for the dashboard ---------------------------------- */
        xSemaphoreTake(s.lock, portMAX_DELAY);
        memcpy(s.mhi, mhi, MHI_PIXELS);
        s.motion_energy = mhi_ctx.motion_energy;
        s.frames++;
        xSemaphoreGive(s.lock);

        /* --- dataset capture state machine ------------------------------ */
        xSemaphoreTake(s.lock, portMAX_DELAY);
        bool finish = false;
        if (s.rec_state == REC_ARMED &&
            mhi_ctx.motion_energy >= MOTION_TRIGGER_PIXELS) {
            /* Motion just started. Seed with the pre-roll already in the ring
             * so the onset of the gesture is captured, then collect the rest. */
            rec_ring_snapshot(s.rec_buf);
            s.rec_collected = (s.rec_filled < REC_PREROLL)
                                  ? s.rec_filled : REC_PREROLL;
            s.rec_state = REC_CAPTURING;
        }
        if (s.rec_state == REC_CAPTURING) {
            /* rec_ring_push() appended the frame just now, at head-1. */
            const uint8_t idx = (uint8_t) ((s.rec_head + MHI_SEQ_LEN - 1u)
                                           % MHI_SEQ_LEN);
            memcpy(s.rec_buf + (size_t) s.rec_collected * MHI_PIXELS,
                   s.rec_ring + (size_t) idx * MHI_PIXELS, MHI_PIXELS);
            s.rec_collected++;
            if (s.rec_collected >= MHI_SEQ_LEN) {
                s.rec_state = REC_DONE;
                finish = true;
            }
        }
        xSemaphoreGive(s.lock);

        if (finish) {
            xSemaphoreGive(s.rec_sem);
        }

        esp_camera_fb_return(fb);

        /* --- fps accounting --------------------------------------------- */
        s_fps_frames++;
        const uint32_t now = (uint32_t) (esp_timer_get_time() / 1000);
        if (now - s_fps_t0 >= 1000) {
            const uint32_t milli = (s_fps_frames * 1000u) / (now - s_fps_t0);
            xSemaphoreTake(s.lock, portMAX_DELAY);
            s.fps_milli = milli;
            xSemaphoreGive(s.lock);
            ESP_LOGI(TAG, "fps=%.1f infer=%ums motion=%u",
                     milli / 1000.0f, (unsigned) s.last_infer_ms,
                     (unsigned) mhi_ctx.motion_energy);
            s_fps_t0 = now;
            s_fps_frames = 0;
        }
    }
}

/* ======================================================================== */
/* app_main                                                                  */
/* ======================================================================== */

void app_main(void)
{
    ESP_LOGI(TAG, "=== ESP32-CAM sign detector ===");

    esp_err_t nvs = nvs_flash_init();
    if (nvs == ESP_ERR_NVS_NO_FREE_PAGES || nvs == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        nvs_flash_init();
    }

    s.lock = xSemaphoreCreateMutex();
    s.rec_sem = xSemaphoreCreateBinary();
    s.rec_ring = heap_caps_malloc(MHI_SEQ_LEN * MHI_PIXELS, MALLOC_CAP_SPIRAM);
    s.rec_buf = heap_caps_malloc(RECORD_SAMPLE_BYTES, MALLOC_CAP_SPIRAM);
    s.leader_class = -1;
    s.last_class = -1;

    if (s.rec_ring == NULL || s.rec_buf == NULL) {
        ESP_LOGE(TAG, "PSRAM alloc failed. Is your board's PSRAM populated? "
                      "See M0 in README.md");
    } else {
        ESP_LOGI(TAG, "PSRAM ok: ring %d B, record buf %d B",
                 MHI_SEQ_LEN * MHI_PIXELS, RECORD_SAMPLE_BYTES);
    }

    if (!camera_init()) {
        ESP_LOGE(TAG, "camera init failed");
        return;
    }

    s_model_ready = infer_init();
    if (s_model_ready) {
        ESP_LOGI(TAG, "inference enabled");

        /* Run the self-test before anything else uses the model. A failure
         * here means the pipeline is wrong, and it is far cheaper to learn
         * that at boot than from a dashboard that shows confident garbage. */
        infer_selftest_t st;
        if (infer_selftest(&st)) {
            if (st.class_ok) {
                ESP_LOGI(TAG, "selftest PASS  class=%d(%s) score=%.3f "
                              "expected %d(%s) %.3f  [%ums]",
                         st.got_class, infer_label(st.got_class),
                         (double) st.got_score, st.want_class,
                         infer_label(st.want_class), (double) st.want_score,
                         (unsigned) infer_last_ms());
            } else {
                ESP_LOGE(TAG, "selftest FAIL  got class %d(%s) score=%.3f, "
                              "expected %d(%s) %.3f",
                         st.got_class, infer_label(st.got_class),
                         (double) st.got_score, st.want_class,
                         infer_label(st.want_class), (double) st.want_score);
                ESP_LOGE(TAG, "  If the model is int8, this is the known "
                              "quantisation collapse (esp-tflite-micro#108).");
                ESP_LOGE(TAG, "  Otherwise check MHI parity: tools/run_parity.sh");
            }
        } else {
            ESP_LOGW(TAG, "no golden vector; run pc/train.py to generate one");
        }
    }

    wifi_start();
    http_start();

    xTaskCreatePinnedToCore(capture_task, "capture", 8192, NULL, 5, NULL, 1);
    ESP_LOGI(TAG, "running");
}
