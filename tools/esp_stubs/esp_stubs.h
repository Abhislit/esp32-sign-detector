/*
 * esp_stubs.h - minimal stand-ins for the ESP-IDF and camera headers.
 *
 * These exist so src/main.c and src/infer.cpp can be syntax-checked on the host
 * with `gcc -fsyntax-only`, which is the only automated check possible without
 * the ESP-IDF toolchain and a board. main.c is ~700 lines of code that would
 * otherwise have no automated checking at all.
 *
 * They contain no real logic and must never be linked into a firmware build;
 * tools/run_tests.sh only passes them via -I to a -fsyntax-only invocation.
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>

/* ---- stdint helpers ------------------------------------------------------*/
#define IPSTR "%d.%d.%d.%d"
#define IP2STR(a) (int) (((uint32_t) (a)) & 0xffu), \
                   (int) (((uint32_t) (a) >> 8) & 0xffu), \
                   (int) (((uint32_t) (a) >> 16) & 0xffu), \
                   (int) (((uint32_t) (a) >> 24) & 0xffu)

/* ---- esp_err_t -----------------------------------------------------------*/
typedef int esp_err_t;
#define ESP_OK 0
#define ESP_FAIL -1
#define ESP_ERROR_CHECK(x) do { (void) (x); } while (0)
#define ESP_ERROR_CHECK_NONE(x) (void) (x)

/* ---- logging -------------------------------------------------------------*/
#define ESP_LOGE(tag, fmt, ...) ((void) (tag), (void) printf(fmt "\n", ##__VA_ARGS__))
#define ESP_LOGW(tag, fmt, ...) ((void) (tag), (void) printf(fmt "\n", ##__VA_ARGS__))
#define ESP_LOGI(tag, fmt, ...) ((void) (tag), (void) printf(fmt "\n", ##__VA_ARGS__))
#define ESP_LOGD(tag, fmt, ...) ((void) (tag), (void) printf(fmt "\n", ##__VA_ARGS__))

/* ---- freertos ------------------------------------------------------------*/
#define portMAX_DELAY 0xffffffffu
#define pdTRUE 1
#define pdFALSE 0
#define pdMS_TO_TICKS(ms) ((unsigned) (ms))
#define configSTACK_DEPTH_TYPE uint32_t
#define MALLOC_CAP_INTERNAL (1 << 11)
#define MALLOC_CAP_SPIRAM (1 << 10)

typedef void *TaskHandle_t;
typedef void *SemaphoreHandle_t;
typedef int BaseType_t;

SemaphoreHandle_t xSemaphoreCreateMutex(void);
SemaphoreHandle_t xSemaphoreCreateBinary(void);
BaseType_t xSemaphoreTake(SemaphoreHandle_t s, unsigned ticks);
BaseType_t xSemaphoreGive(SemaphoreHandle_t s);
BaseType_t xTaskCreatePinnedToCore(void (*fn)(void *), const char *name,
                                   uint32_t stack, void *arg, unsigned prio,
                                   TaskHandle_t *out, int core);
void vTaskDelay(unsigned ticks);

/* ---- esp core ------------------------------------------------------------*/
typedef struct {
    int cores;
    int revision;
    int flash_size;
} chip_info_t;
void esp_chip_info(chip_info_t *out);

int64_t esp_timer_get_time(void);
size_t heap_caps_get_free_size(unsigned caps);
size_t heap_caps_get_total_size(unsigned caps);
void *heap_caps_malloc(size_t n, unsigned caps);
uint32_t esp_spiram_get_size(void);

typedef enum {
    ESP_RST_UNKNOWN = 0,
    ESP_RST_POWERON,
    ESP_RST_EXT,
    ESP_RST_SW,
    ESP_RST_PANIC,
    ESP_RST_INT_WDT,
    ESP_RST_TASK_WDT,
    ESP_RST_WDT,
    ESP_RST_DEEPSLEEP,
    ESP_RST_BROWNOUT,
    ESP_RST_SDIO,
} esp_reset_reason_t;
esp_reset_reason_t esp_reset_reason(void);

typedef enum { ESP_ERR_NVS_BASE = 0x1100, ESP_ERR_NVS_NO_FREE_PAGES = 0x110F,
               ESP_ERR_NVS_NEW_VERSION_FOUND = 0x1110 } esp_err_nvs_t;
esp_err_t nvs_flash_init(void);
esp_err_t nvs_flash_erase(void);

/* ---- wifi ----------------------------------------------------------------*/
typedef enum { WIFI_MODE_NULL = 0, WIFI_MODE_STA, WIFI_MODE_AP,
               WIFI_MODE_APSTA } wifi_mode_t;
typedef enum { WIFI_AUTH_OPEN = 0, WIFI_AUTH_WPA2_PSK } wifi_auth_mode_t;
typedef enum { WIFI_IF_STA = 0, WIFI_IF_AP } wifi_interface_t;

typedef struct {
    uint8_t ssid[32];
    uint8_t password[64];
    struct { wifi_auth_mode_t authmode; } threshold;
} wifi_sta_config_t;
typedef struct { uint8_t ssid[32]; uint8_t password[64];
                 uint8_t max_connection; uint8_t authmode; } wifi_ap_config_t;
typedef union { wifi_sta_config_t sta; wifi_ap_config_t ap; } wifi_config_t;

typedef struct { uint8_t reserved[64]; } wifi_init_config_t;
#define WIFI_INIT_CONFIG_DEFAULT() ((wifi_init_config_t) { { 0 } })

typedef enum { WIFI_EVENT_WIFI_READY = 0, WIFI_EVENT_STA_START,
               WIFI_EVENT_STA_STOP, WIFI_EVENT_STA_CONNECTED,
               WIFI_EVENT_STA_DISCONNECTED } wifi_event_t;
typedef enum { IP_EVENT_STA_GOT_IP = 0, IP_EVENT_STA_LOST_IP } ip_event_t;
typedef struct { uint32_t ip, netmask, gw; } esp_netif_ip_info_t;
typedef struct { esp_netif_ip_info_t ip_info; } ip_event_got_ip_t;

typedef void *esp_event_base_t;
#define WIFI_EVENT ((esp_event_base_t) "WIFI_EVENT")
#define IP_EVENT  ((esp_event_base_t) "IP_EVENT")
#define ESP_EVENT_ANY_ID (-1)

esp_err_t esp_wifi_init(const wifi_init_config_t *cfg);
esp_err_t esp_wifi_set_mode(wifi_mode_t m);
esp_err_t esp_wifi_set_config(wifi_interface_t i, wifi_config_t *c);
esp_err_t esp_wifi_start(void);
esp_err_t esp_wifi_connect(void);
esp_err_t esp_event_handler_register(esp_event_base_t b, int32_t id,
                                     void (*h)(void *, esp_event_base_t, int32_t, void *),
                                     void *arg);

/* ---- camera --------------------------------------------------------------*/
#define LEDC_TIMER_0 0
#define LEDC_CHANNEL_0 0

typedef enum {
    PIXFORMAT_RGB565 = 0, PIXFORMAT_JPEG, PIXFORMAT_GRAYSCALE,
} pixformat_t;
typedef enum { CAMERA_FB_IN_PSRAM = 0, CAMERA_FB_IN_SRAM } camera_fb_location_t;
typedef enum { CAMERA_GRAB_WHEN_EMPTY = 0, CAMERA_GRAB_LATEST } camera_grab_mode_t;

typedef enum {
    FRAMESIZE_QQVGA = 0, FRAMESIZE_QVGA, FRAMESIZE_CIF, FRAMESIZE_HVGA,
    FRAMESIZE_VGA, FRAMESIZE_SVGA, FRAMESIZE_XGA, FRAMESIZE_SXGA, FRAMESIZE_UXGA,
} framesize_t;

typedef struct {
    int pwdn, reset, xclk, sccb_sda, sccb_scl;
    int d7, d6, d5, d4, d3, d2, d1, d0;
    int vsync, href, pclk;
} camera_pins_t;

typedef struct {
    int pin_pwdn, pin_reset, pin_xclk;
    int pin_sccb_sda, pin_sccb_scl;
    int pin_d7, pin_d6, pin_d5, pin_d4, pin_d3, pin_d2, pin_d1, pin_d0;
    int pin_vsync, pin_href, pin_pclk;
    int xclk_freq_hz;
    int ledc_timer, ledc_channel;
    framesize_t frame_size;
    int jpeg_quality;
    int fb_count;
    camera_fb_location_t fb_location;
    camera_grab_mode_t grab_mode;
} camera_config_t;

typedef struct { int width, height; } frame_size_t;

typedef struct {
    struct { uint16_t PID, VER; } id;
    frame_size_t *cur_size;
    int (*set_quality)(void *s, int q);
    int (*set_vflip)(void *s, int on);
    int (*set_hmirror)(void *s, int on);
} sensor_t;

#define OV2640_PID 0x26

typedef struct {
    uint8_t *buf;
    size_t len;
    size_t width, height;
    pixformat_t format;
} camera_fb_t;

esp_err_t esp_camera_init(const camera_config_t *cfg);
camera_fb_t *esp_camera_fb_get(void);
void esp_camera_fb_return(camera_fb_t *fb);
sensor_t *esp_camera_sensor_get(void);
esp_err_t esp_camera_set_framesize(sensor_t *s, framesize_t size);

esp_err_t fmt2jpg(const uint8_t *src, size_t w, size_t h, pixformat_t fmt,
                  uint8_t quality, uint8_t **dst, size_t *dst_len);

/* ---- http server ---------------------------------------------------------*/
typedef void *httpd_handle_t;
typedef struct httpd_req *httpd_req_t;
typedef int httpd_method_t;
#define HTTP_GET 1
#define HTTPD_500_INTERNAL_SERVER_ERROR 500
#define HTTPD_RESP_USE_STRLEN (-1)

typedef struct {
    const char *uri;
    httpd_method_t method;
    esp_err_t (*handler)(httpd_req_t *r);
    void *user_ctx;
} httpd_uri_t;

typedef struct {
    unsigned server_port;
    unsigned ctrl_port;
    size_t max_open_sockets;
    size_t max_uri_handlers;
    size_t max_resp_headers;
    size_t stack_size;
    bool lru_purge_enable;
    bool uri_match_fn;
} httpd_config_t;
#define HTTPD_DEFAULT_CONFIG() (httpd_config_t) { 0 }

esp_err_t httpd_start(httpd_handle_t *h, const httpd_config_t *cfg);
esp_err_t httpd_register_uri_handler(httpd_handle_t h, const httpd_uri_t *u);
esp_err_t httpd_resp_send(httpd_req_t *r, const char *buf, ssize_t len);
esp_err_t httpd_resp_send_chunk(httpd_req_t *r, const char *buf, ssize_t len);
esp_err_t httpd_resp_set_type(httpd_req_t *r, const char *t);
esp_err_t httpd_resp_set_hdr(httpd_req_t *r, const char *k, const char *v);
esp_err_t httpd_resp_send_err(httpd_req_t *r, int code, const char *msg);
int httpd_req_get_url_query_len(httpd_req_t *r);
int httpd_req_get_url_query_str(httpd_req_t *r, char *buf, int len);
