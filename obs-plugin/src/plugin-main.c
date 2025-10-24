// plugin-main.c - OBS plugin to launch service and stream mixed audio
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <obs-module.h>
#include <obs.h>
#include <obs-frontend-api.h>
#include <media-io/audio-io.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

#pragma comment(lib, "ws2_32.lib")

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("autoaffili", "en-US")

// Forward decls
static SOCKET connect_to_port(int port);

MODULE_EXPORT const char *obs_module_name(void)
{
    return "Autoaffili";
}

MODULE_EXPORT const char *obs_module_description(void)
{
    return "Streams OBS mixed audio to the Autoaffili service and launches the companion helper.";
}

#define AUDIO_BRIDGE_HOST "127.0.0.1"
static int g_port_desktop = 8765;
static int g_port_mic = 8766;
static int g_mix_idx_desktop = 0;
static int g_mix_idx_mic = 1;

static PROCESS_INFORMATION g_procInfo = {0};
static volatile SOCKET g_audio_sock_desktop = INVALID_SOCKET;
static volatile SOCKET g_audio_sock_mic = INVALID_SOCKET;
static HANDLE g_connect_thread = NULL;
static CRITICAL_SECTION g_sock_lock;
static volatile bool g_shutdown = false;
static volatile bool g_exit_signaled = false;
static volatile bool g_streaming_active = false;

static void handle_socket_failure(void);
static void source_audio_cb(void *param, obs_source_t *source, const struct audio_data *audio, bool muted);
static void signal_service_shutdown_event(void);

// ------------------------------------------------------------
// Diagnostics helpers
// ------------------------------------------------------------
// Source-level capture (per-category)
static bool g_source_cb_registered_desktop = false;
static bool g_source_cb_registered_mic = false;
static obs_source_t *g_src_desktop = NULL;
static obs_source_t *g_src_mic = NULL;
static int g_tag_desktop = 0;
static int g_tag_mic = 1;
static bool g_logged_flow_desktop = false;
static bool g_logged_flow_mic = false;
static float *g_interleave_buffer = NULL;
static size_t g_interleave_capacity = 0;
static uint32_t g_audio_channels = 0;
static uint32_t g_audio_sample_rate = 48000;
static bool g_logged_enum_once = false;
static ULONGLONG g_last_flow_mic_ms = 0;
static const ULONGLONG MIC_INACTIVITY_MS = 20000ULL; // 240 seconds

static void load_config(void);

// (frontend Tools menu integration removed per request)

struct audio_packet_header {
    uint32_t frames;
    uint32_t channels;
    uint32_t sample_rate;
};

static void close_audio_socket_locked(void)
{
    if (g_audio_sock_desktop != INVALID_SOCKET) { closesocket(g_audio_sock_desktop); g_audio_sock_desktop = INVALID_SOCKET; }
    if (g_audio_sock_mic != INVALID_SOCKET) { closesocket(g_audio_sock_mic); g_audio_sock_mic = INVALID_SOCKET; }
}

static void handle_socket_failure(void)
{
    EnterCriticalSection(&g_sock_lock);
    close_audio_socket_locked();
    LeaveCriticalSection(&g_sock_lock);
}

static void free_interleave_buffer(void)
{
    if (g_interleave_buffer) {
        free(g_interleave_buffer);
        g_interleave_buffer = NULL;
        g_interleave_capacity = 0;
    }
}

static inline bool is_desktop_type(const char *id)
{
    if (!id) return false;
    // Windows desktop output + app capture + common desktop-like sources
    return strcmp(id, "wasapi_output_capture") == 0 ||
           strcmp(id, "application_audio_capture") == 0 ||
           strcmp(id, "ffmpeg_source") == 0 ||
           strcmp(id, "browser_source") == 0 ||
           strcmp(id, "vlc_source") == 0;
}

static inline bool is_mic_type(const char *id)
{
    if (!id) return false;
    // Windows mic input + capture card devices with audio
    return strcmp(id, "wasapi_input_capture") == 0 ||
           strcmp(id, "dshow_input") == 0;
}

struct enum_log_ctx { const char *phase; };
static bool enum_log_cb(void *param, obs_source_t *src)
{
    struct enum_log_ctx *ctx = (struct enum_log_ctx *)param;
    const char *id = obs_source_get_id(src);
    const char *name = obs_source_get_name(src);
    bool active = obs_source_active(src);
    bool is_desk = is_desktop_type(id);
    bool is_mic = is_mic_type(id);
    blog(LOG_INFO, "autoaffili: sources[%s] id=%s name=%s active=%d desk=%d mic=%d",
         ctx && ctx->phase ? ctx->phase : "", id ? id : "", name ? name : "", (int)active, (int)is_desk, (int)is_mic);
    return true;
}

static void log_all_sources_once(const char *phase)
{
    if (g_logged_enum_once) return;
    struct enum_log_ctx ctx = { phase };
    obs_enum_all_sources(enum_log_cb, &ctx);
    g_logged_enum_once = true;
}

static void signal_service_shutdown_event(void)
{
    HANDLE h = OpenEventA(EVENT_MODIFY_STATE, FALSE, "Local\\autoaffili_shutdown");
    if (h) {
        SetEvent(h);
        CloseHandle(h);
    }
}

static void log_sentinel_state(const char *phase)
{
    const char *appdata = getenv("APPDATA");
    if (!appdata) {
        blog(LOG_INFO, "autoaffili: [%s] APPDATA not set; cannot inspect .sentinel", phase ? phase : "?");
        return;
    }
    char dir[MAX_PATH];
    snprintf(dir, MAX_PATH, "%s\\obs-studio\\.sentinel", appdata);
    DWORD attrs = GetFileAttributesA(dir);
    if (attrs == INVALID_FILE_ATTRIBUTES || !(attrs & FILE_ATTRIBUTE_DIRECTORY)) {
        blog(LOG_INFO, "autoaffili: [%s] .sentinel missing (%s)", phase ? phase : "?", dir);
        return;
    }
    char pattern[MAX_PATH];
    snprintf(pattern, MAX_PATH, "%s\\run_*", dir);
    WIN32_FIND_DATAA ffd; ZeroMemory(&ffd, sizeof(ffd));
    HANDLE h = FindFirstFileA(pattern, &ffd);
    int count = 0;
    if (h != INVALID_HANDLE_VALUE) {
        do {
            if (!(ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
                count++;
        } while (FindNextFileA(h, &ffd));
        FindClose(h);
    }
    blog(LOG_INFO, "autoaffili: [%s] .sentinel run_* count=%d", phase ? phase : "?", count);
}

static void on_frontend_event(enum obs_frontend_event event, void *priv)
{
    UNUSED_PARAMETER(priv);
    if (event == OBS_FRONTEND_EVENT_STREAMING_STARTED) {
        g_streaming_active = true;
        blog(LOG_INFO, "autoaffili: OBS streaming started; enabling audio forward");
    } else if (event == OBS_FRONTEND_EVENT_STREAMING_STOPPED) {
        g_streaming_active = false;
        blog(LOG_INFO, "autoaffili: OBS streaming stopped; disabling audio forward");
    } else if (event == OBS_FRONTEND_EVENT_EXIT) {
        blog(LOG_INFO, "autoaffili: OBS_FRONTEND_EVENT_EXIT received; signaling shutdown");
        g_exit_signaled = true;
        g_shutdown = true;
        signal_service_shutdown_event();
        handle_socket_failure();
        if (g_source_cb_registered_desktop && g_src_desktop) {
            obs_source_remove_audio_capture_callback(g_src_desktop, source_audio_cb, &g_tag_desktop);
            obs_source_release(g_src_desktop);
            g_src_desktop = NULL;
            g_source_cb_registered_desktop = false;
        }
        if (g_source_cb_registered_mic && g_src_mic) {
            obs_source_remove_audio_capture_callback(g_src_mic, source_audio_cb, &g_tag_mic);
            obs_source_release(g_src_mic);
            g_src_mic = NULL;
            g_source_cb_registered_mic = false;
        }
    }
}

static void source_audio_cb(void *param, obs_source_t *source, const struct audio_data *audio, bool muted)
{
    if (!audio || audio->frames == 0 || muted)
        return;
    // Gate audio forwarding until streaming is active
    if (!g_streaming_active)
        return;

    int tag = 0;
    if (param == &g_tag_mic) tag = 1;

    SOCKET sock = INVALID_SOCKET;
    EnterCriticalSection(&g_sock_lock);
    sock = (tag == 1) ? g_audio_sock_mic : g_audio_sock_desktop;
    LeaveCriticalSection(&g_sock_lock);
    if (sock == INVALID_SOCKET)
        return;

    // Log first packet per stream to confirm flow
    if (tag == 1) {
        if (!g_logged_flow_mic) {
            const char *n = obs_source_get_name(source);
            blog(LOG_INFO, "autoaffili: mic audio flowing: src=%s", n ? n : "");
            g_logged_flow_mic = true;
        }
    } else {
        if (!g_logged_flow_desktop) {
            const char *n = obs_source_get_name(source);
            blog(LOG_INFO, "autoaffili: desktop audio flowing: src=%s", n ? n : "");
            g_logged_flow_desktop = true;
        }
    }

    // Log first packet per stream to confirm flow
    if (tag == 1) {
        if (!g_logged_flow_mic) {
            const char *n = obs_source_get_name(source);
            blog(LOG_INFO, "autoaffili: mic audio flowing: src=%s", n ? n : "");
            g_logged_flow_mic = true;
        }
    } else {
        if (!g_logged_flow_desktop) {
            const char *n = obs_source_get_name(source);
            blog(LOG_INFO, "autoaffili: desktop audio flowing: src=%s", n ? n : "");
            g_logged_flow_desktop = true;
        }
    }

    // Update last-flow timestamps
    ULONGLONG now_ms = GetTickCount64();
    if (tag == 1) g_last_flow_mic_ms = now_ms;

    // Determine channel count from source/type
    int planes_non_null = 0;
    for (uint32_t ch = 0; ch < MAX_AV_PLANES; ++ch) {
        if (audio->data[ch]) planes_non_null++;
    }
    const char *sid = obs_source_get_id(source);
    uint32_t channels = 0;
    if (planes_non_null > 1) {
        channels = (uint32_t)planes_non_null;
    } else {
        // Interleaved; guess sensible defaults per source type
        if (sid && strcmp(sid, "wasapi_input_capture") == 0)
            channels = 1; // typical mic
        else if (sid && strcmp(sid, "dshow_input") == 0)
            channels = 1; // many camera mics are mono
        else if (sid && strcmp(sid, "wasapi_output_capture") == 0)
            channels = 2; // typical desktop
        else
            channels = g_audio_channels ? g_audio_channels : 2;
    }
    if (channels == 0) channels = 2;

    size_t required = (size_t)audio->frames * (size_t)channels;
    if (required == 0)
        return;

    if (required > g_interleave_capacity) {
        float *tmp = (float *)realloc(g_interleave_buffer, required * sizeof(float));
        if (!tmp)
            return;
        g_interleave_buffer = tmp;
        g_interleave_capacity = required;
    }

    // Detect layout: if only data[0] is non-null, treat as interleaved; if multiple planes present, treat as planar
    bool planar = (planes_non_null > 1);

    if (!planar) {
        // Interleaved float32 [frames * channels]
        const float *src = (const float *)(audio->data[0]);
        if (src) {
            memcpy(g_interleave_buffer, src, required * sizeof(float));
        } else {
            // Fallback to zeros if no data
            memset(g_interleave_buffer, 0, required * sizeof(float));
        }
    } else {
        // Planar: data[ch] points to float[frames]
        for (uint32_t ch = 0; ch < channels; ++ch) {
            const float *src = (const float *)(audio->data[ch]);
            if (!src) {
                for (uint32_t i = 0; i < audio->frames; ++i)
                    g_interleave_buffer[i * channels + ch] = 0.0f;
                continue;
            }
            for (uint32_t i = 0; i < audio->frames; ++i)
                g_interleave_buffer[i * channels + ch] = src[i];
        }
    }

    struct audio_packet_header header;
    header.frames = audio->frames;
    header.channels = channels;
    header.sample_rate = g_audio_sample_rate ? g_audio_sample_rate : 48000;

    if (send(sock, (const char *)&header, sizeof(header), 0) == SOCKET_ERROR) {
        handle_socket_failure();
        return;
    }

    const char *payload = (const char *)g_interleave_buffer;
    int total_bytes = (int)(required * sizeof(float));
    int offset = 0;
    while (offset < total_bytes) {
        int rc = send(sock, payload + offset, total_bytes - offset, 0);
        if (rc == SOCKET_ERROR) {
            handle_socket_failure();
            return;
        }
        offset += rc;
    }
}

static bool enum_pick_desktop(void *param, obs_source_t *src)
{
    UNUSED_PARAMETER(param);
    if (g_src_desktop)
        return true; // already picked
    const char *id = obs_source_get_id(src);
    if (!id || !is_desktop_type(id))
        return true;
    // Prefer active sources to avoid attaching to inactive devices
    if (!obs_source_active(src))
        return true;
    g_src_desktop = obs_source_get_ref(src);
    obs_source_add_audio_capture_callback(g_src_desktop, source_audio_cb, &g_tag_desktop);
    g_source_cb_registered_desktop = true;
    {
        const char *name = obs_source_get_name(src);
        blog(LOG_INFO, "autoaffili: attached desktop source id=%s name=%s", id, name ? name : "");
    }
    return true;
}

static bool enum_pick_mic(void *param, obs_source_t *src)
{
    UNUSED_PARAMETER(param);
    if (g_src_mic)
        return true; // already picked first matching mic
    const char *id = obs_source_get_id(src);
    if (!id || !is_mic_type(id))
        return true;
    // Prefer active sources to avoid attaching to inactive devices
    if (!obs_source_active(src))
        return true;
    g_src_mic = obs_source_get_ref(src);
    obs_source_add_audio_capture_callback(g_src_mic, source_audio_cb, &g_tag_mic);
    g_source_cb_registered_mic = true;
    {
        const char *name = obs_source_get_name(src);
        blog(LOG_INFO, "autoaffili: attached mic source id=%s name=%s", id, name ? name : "");
    }
    return true;
}

static void attach_sources_once(void)
{
    // Pick the first matching desktop/mic source present at load time
    obs_enum_all_sources(enum_pick_desktop, NULL);
    obs_enum_all_sources(enum_pick_mic, NULL);
    log_all_sources_once("attach");
}

struct mic_pick_ctx { obs_source_t *current; obs_source_t *prefer; obs_source_t *fallback; };
static bool enum_pick_next_mic(void *param, obs_source_t *src)
{
    struct mic_pick_ctx *ctx = (struct mic_pick_ctx *)param;
    const char *id = obs_source_get_id(src);
    if (!id || !is_mic_type(id) || !obs_source_active(src))
        return true;
    if (ctx->current && src == ctx->current)
        return true; // skip current
    if (strcmp(id, "wasapi_input_capture") == 0) {
        if (!ctx->prefer) ctx->prefer = obs_source_get_ref(src);
    } else {
        if (!ctx->fallback) ctx->fallback = obs_source_get_ref(src);
    }
    return true;
}

static void reattach_mic_if_inactive(void)
{
    if (!g_src_mic)
        return;
    ULONGLONG now_ms = GetTickCount64();
    if (g_last_flow_mic_ms == 0)
        g_last_flow_mic_ms = now_ms; // seed
    ULONGLONG delta = now_ms - g_last_flow_mic_ms;
    if (delta < MIC_INACTIVITY_MS)
        return;
    // Find next best active mic candidate different from current
    struct mic_pick_ctx ctx; ZeroMemory(&ctx, sizeof(ctx)); ctx.current = g_src_mic;
    obs_enum_all_sources(enum_pick_next_mic, &ctx);
    obs_source_t *next = ctx.prefer ? ctx.prefer : ctx.fallback;
    if (next) {
        const char *oldname = obs_source_get_name(g_src_mic);
        obs_source_remove_audio_capture_callback(g_src_mic, source_audio_cb, &g_tag_mic);
        obs_source_release(g_src_mic);
        g_src_mic = next;
        obs_source_add_audio_capture_callback(g_src_mic, source_audio_cb, &g_tag_mic);
        g_source_cb_registered_mic = true;
        const char *newid = obs_source_get_id(g_src_mic);
        const char *newname = obs_source_get_name(g_src_mic);
        blog(LOG_INFO, "autoaffili: mic inactive %.1fs; switched from '%s' to id=%s name=%s",
             (double)delta / 1000.0, oldname ? oldname : "", newid ? newid : "", newname ? newname : "");
        g_last_flow_mic_ms = now_ms;
    }
}

static bool start_service_exe(void)
{
    // If a service is already running (ports open), reuse it
    SOCKET test0 = connect_to_port(g_port_desktop);
    SOCKET test1 = connect_to_port(g_port_mic);
    if (test0 != INVALID_SOCKET || test1 != INVALID_SOCKET) {
        if (test0 != INVALID_SOCKET) closesocket(test0);
        if (test1 != INVALID_SOCKET) closesocket(test1);
        blog(LOG_INFO, "autoaffili: service already running; not launching new instance");
        return true;
    }

    char pathProgramData[MAX_PATH] = {0};
    char pathLocalApp[MAX_PATH] = {0};
    char buf[MAX_PATH] = {0};
    if (GetEnvironmentVariableA("PROGRAMDATA", buf, MAX_PATH)) {
        snprintf(pathProgramData, MAX_PATH, "%s\\autoaffili\\autoaffili.exe", buf);
    }
    if (GetEnvironmentVariableA("LOCALAPPDATA", buf, MAX_PATH)) {
        snprintf(pathLocalApp, MAX_PATH, "%s\\autoaffili\\autoaffili.exe", buf);
    }

    const char *candidates[3] = {0};
    int ci = 0;
    if (pathProgramData[0]) candidates[ci++] = pathProgramData;
    if (pathLocalApp[0]) candidates[ci++] = pathLocalApp;

    STARTUPINFOA si;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&g_procInfo, sizeof(g_procInfo));

    for (int i = 0; i < ci; ++i) {
        const char *p = candidates[i];
        BOOL ok = CreateProcessA(
            p,
            NULL,
            NULL,
            NULL,
            FALSE,
            CREATE_NO_WINDOW,
            NULL,
            NULL,
            &si,
            &g_procInfo
        );
        if (ok) {
            blog(LOG_INFO, "autoaffili: launched (pid=%u) from %s.", (unsigned)g_procInfo.dwProcessId, p);
            return true;
        }
        DWORD err = GetLastError();
        blog(LOG_INFO, "autoaffili: launch failed (err=%u) at %s", err, p);
        ZeroMemory(&g_procInfo, sizeof(g_procInfo));
    }
    blog(LOG_WARNING, "autoaffili: Failed to launch helper from ProgramData and LocalAppData");
    return false;
}

static void stop_service_exe(void)
{
    if (g_procInfo.hProcess) {
        TerminateProcess(g_procInfo.hProcess, 0);
        CloseHandle(g_procInfo.hProcess);
        CloseHandle(g_procInfo.hThread);
        ZeroMemory(&g_procInfo, sizeof(g_procInfo));
        blog(LOG_INFO, "autoaffili: terminated.");
    }
}

static SOCKET connect_to_port(int port)
{
    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == INVALID_SOCKET) return INVALID_SOCKET;
    struct sockaddr_in addr; ZeroMemory(&addr, sizeof(addr));
    addr.sin_family = AF_INET; addr.sin_port = htons((u_short)port);
    inet_pton(AF_INET, AUDIO_BRIDGE_HOST, &addr.sin_addr);
    if (connect(s, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        int err = WSAGetLastError();
        blog(LOG_INFO, "autoaffili: connect failed port=%d err=%d", port, err);
        closesocket(s); return INVALID_SOCKET;
    }
    return s;
}

static DWORD WINAPI connect_thread_proc(LPVOID param)
{
    UNUSED_PARAMETER(param);

    while (!g_shutdown) {
        SOCKET s0 = connect_to_port(g_port_desktop);
        SOCKET s1 = connect_to_port(g_port_mic);
        if (s0 != INVALID_SOCKET || s1 != INVALID_SOCKET) {
            blog(LOG_INFO, "autoaffili: audio bridge connected (desktop:%s@%d, mic:%s@%d)", s0!=INVALID_SOCKET?"ok":"fail", g_port_desktop, s1!=INVALID_SOCKET?"ok":"fail", g_port_mic);
            EnterCriticalSection(&g_sock_lock);
            if (s0 != INVALID_SOCKET) g_audio_sock_desktop = s0;
            if (s1 != INVALID_SOCKET) g_audio_sock_mic = s1;
            LeaveCriticalSection(&g_sock_lock);

            audio_t *audio = obs_get_audio();
            if (audio) {
                g_audio_channels = (uint32_t)audio_output_get_channels(audio);
                g_audio_sample_rate = (uint32_t)audio_output_get_sample_rate(audio);
                const struct audio_output_info *info = audio_output_get_info(audio);
            }
            // Attach to sources (first matching per category)
            attach_sources_once();
            while (!g_shutdown) {
                EnterCriticalSection(&g_sock_lock);
                SOCKET a = g_audio_sock_desktop;
                SOCKET b = g_audio_sock_mic;
                LeaveCriticalSection(&g_sock_lock);
                // Retry attachment if sources not yet found (handles sources added after plugin load)
                if (!g_src_desktop || !g_src_mic) {
                    attach_sources_once();
                }
                // Mic flow watchdog: switch if inactive for too long
                reattach_mic_if_inactive();
                // Attempt to (re)connect any missing socket while the other is up
                if (a == INVALID_SOCKET) {
                    SOCKET s = connect_to_port(g_port_desktop);
                    if (s != INVALID_SOCKET) {
                        EnterCriticalSection(&g_sock_lock);
                        g_audio_sock_desktop = s;
                        LeaveCriticalSection(&g_sock_lock);
                        blog(LOG_INFO, "autoaffili: desktop bridge reconnected @%d", g_port_desktop);
                    }
                }
                if (b == INVALID_SOCKET) {
                    SOCKET s = connect_to_port(g_port_mic);
                    if (s != INVALID_SOCKET) {
                        EnterCriticalSection(&g_sock_lock);
                        g_audio_sock_mic = s;
                        LeaveCriticalSection(&g_sock_lock);
                        blog(LOG_INFO, "autoaffili: mic bridge reconnected @%d", g_port_mic);
                    }
                }
                if (a == INVALID_SOCKET && b == INVALID_SOCKET)
                    break;
                Sleep(200);
            }

            // no disconnects for source callbacks here; handled on unload

            EnterCriticalSection(&g_sock_lock);
            close_audio_socket_locked();
            LeaveCriticalSection(&g_sock_lock);
        }

        if (!g_shutdown)
            Sleep(500);
    }

    return 0;
}

MODULE_EXPORT bool obs_module_load(void)
{
    blog(LOG_INFO, "autoaffili plugin loaded. Starting bundled service...");
    log_sentinel_state("load-begin");

    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        blog(LOG_ERROR, "autoaffili: WSAStartup failed");
        return false;
    }

    InitializeCriticalSection(&g_sock_lock);
    load_config();
    blog(LOG_INFO, "autoaffili: using ports desktop=%d mic=%d", g_port_desktop, g_port_mic);
    log_all_sources_once("load");
    // Eagerly attach to first matching sources
    attach_sources_once();

    if (!start_service_exe()) {
        DeleteCriticalSection(&g_sock_lock);
        WSACleanup();
        return false;
    }

    obs_frontend_add_event_callback(on_frontend_event, NULL);

    g_shutdown = false;
    g_connect_thread = CreateThread(NULL, 0, connect_thread_proc, NULL, 0, NULL);
    if (!g_connect_thread) {
        blog(LOG_ERROR, "autoaffili: failed to start connect thread");
        g_shutdown = true;
        stop_service_exe();
        DeleteCriticalSection(&g_sock_lock);
        WSACleanup();
        return false;
    }

    return true;
}

MODULE_EXPORT void obs_module_unload(void)
{
    blog(LOG_INFO, "autoaffili plugin unloading. Cleaning up...");
    obs_frontend_remove_event_callback(on_frontend_event, NULL);
    log_sentinel_state("unload-begin");

    g_shutdown = true;

    if (g_connect_thread) {
        DWORD wr = WaitForSingleObject(g_connect_thread, 5000);
        if (wr == WAIT_TIMEOUT) {
            blog(LOG_WARNING, "autoaffili: connect thread did not exit within 5s during unload; continuing");
        }
        CloseHandle(g_connect_thread);
        g_connect_thread = NULL;
    }

    handle_socket_failure();

    if (g_source_cb_registered_desktop && g_src_desktop) {
        obs_source_remove_audio_capture_callback(g_src_desktop, source_audio_cb, &g_tag_desktop);
        obs_source_release(g_src_desktop);
        g_src_desktop = NULL;
        g_source_cb_registered_desktop = false;
    }
    if (g_source_cb_registered_mic && g_src_mic) {
        obs_source_remove_audio_capture_callback(g_src_mic, source_audio_cb, &g_tag_mic);
        obs_source_release(g_src_mic);
        g_src_mic = NULL;
        g_source_cb_registered_mic = false;
    }

    free_interleave_buffer();
    DeleteCriticalSection(&g_sock_lock);
    WSACleanup();

    // Signal helper to exit even if not launched by this plugin instance
    signal_service_shutdown_event();
    // If we launched it, terminate our child process
    stop_service_exe();
    Sleep(600);
    log_sentinel_state("unload-end");
    blog(LOG_INFO, "autoaffili: unload complete");
}
static void load_config(void)
{
    char buf[64];
    DWORD n;
    n = GetEnvironmentVariableA("OBS_SL_PORT_DESKTOP", buf, sizeof(buf));
    if (n > 0 && n < sizeof(buf)) g_port_desktop = atoi(buf);
    n = GetEnvironmentVariableA("OBS_SL_PORT_MIC", buf, sizeof(buf));
    if (n > 0 && n < sizeof(buf)) g_port_mic = atoi(buf);
    n = GetEnvironmentVariableA("OBS_SL_MIX_DESKTOP", buf, sizeof(buf));
    if (n > 0 && n < sizeof(buf)) g_mix_idx_desktop = atoi(buf);
    n = GetEnvironmentVariableA("OBS_SL_MIX_MIC", buf, sizeof(buf));
    if (n > 0 && n < sizeof(buf)) g_mix_idx_mic = atoi(buf);

    // Optional plugin.ini at %LOCALAPPDATA%\autoaffili\plugin.ini with lines:
    // desktop_port=8765\nmic_port=8766\ndesktop_mix=0\nmic_mix=1
    char appDir[MAX_PATH];
    if (GetEnvironmentVariableA("LOCALAPPDATA", appDir, MAX_PATH)) {
        char path[MAX_PATH];
        snprintf(path, MAX_PATH, "%s\\autoaffili\\plugin.ini", appDir);
        FILE *f = fopen(path, "r");
        if (f) {
            char line[256];
            while (fgets(line, sizeof(line), f)) {
                char *eq = strchr(line, '=');
                if (!eq) continue;
                *eq = '\0';
                char *key = line;
                char *val = eq + 1;
                // trim
                while (*key == ' ' || *key == '\t') key++;
                char *end = val + strlen(val);
                while (end > val && (end[-1] == '\r' || end[-1] == '\n' || end[-1] == ' ' || end[-1] == '\t')) { end[-1] = '\0'; end--; }
                if (strcmp(key, "desktop_port") == 0) g_port_desktop = atoi(val);
                else if (strcmp(key, "mic_port") == 0) g_port_mic = atoi(val);
                else if (strcmp(key, "desktop_mix") == 0) g_mix_idx_desktop = atoi(val);
                else if (strcmp(key, "mic_mix") == 0) g_mix_idx_mic = atoi(val);
            }
            fclose(f);
        }
    }
}

