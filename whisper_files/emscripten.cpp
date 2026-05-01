#include "ggml.h"
#include "whisper.h"

#include <emscripten.h>
#include <emscripten/bind.h>

#include <atomic>
#include <cmath>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

constexpr int N_THREAD = 8;

std::vector<struct whisper_context *> g_contexts(4, nullptr);

std::mutex g_mutex;
std::thread g_worker;

std::atomic<bool> g_running(false);

std::string g_status        = "";
std::string g_status_forced = "";
std::string g_transcribed   = "";
float g_confidence = 0.0f;
float g_pt = 0.0f;

// Per-token data for the last transcribed segment.
// t0/t1 are in centiseconds (whisper internal unit); multiply by 10 for ms.
struct TokenData {
    std::string text;
    float       p;
    int64_t     t0;
    int64_t     t1;
};
std::vector<TokenData> g_tokens;

// Runtime-configurable inference parameters (set via JS setters before/between calls)
int   g_max_tokens = 64;   // raised default for polysynthetic languages (Cree verbs can be 10+ morphemes)
float g_vad_thold  = 0.6f; // VAD speech probability threshold (wparams.vad_params.threshold)
                            // Note: VAD only activates when wparams.vad=true AND a vad_model_path is set

std::vector<float> g_pcmf32;

void stream_set_status(const std::string & status) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_status = status;
}

void stream_main(size_t index, const std::string & lang) {
    stream_set_status("loading data ...");

    // Sampling strategy is greedy for lowest latency; beam search is significantly slower.
    struct whisper_full_params wparams = whisper_full_default_params(whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY);
    bool is_multilingual = whisper_is_multilingual(g_contexts[index]);

    wparams.n_threads        = std::min(N_THREAD, (int) std::thread::hardware_concurrency());
    wparams.offset_ms        = 0;
    wparams.translate        = false;
    wparams.no_context       = true;
    wparams.single_segment   = true;
    wparams.print_realtime   = false;
    wparams.print_progress   = false;
    wparams.print_timestamps = true;
    wparams.print_special    = false;
    wparams.audio_ctx         = 768;  // partial encoder context for better performance
    wparams.temperature_inc   = -1.0f; // disable temperature fallback
    wparams.token_timestamps  = true;  // populate t0/t1 in whisper_token_data
    wparams.language         = is_multilingual ? lang.c_str() : "en";

    printf("stream: using %d threads\n", wparams.n_threads);

    std::vector<float> pcmf32;

    auto & ctx = g_contexts[index];

    const int64_t window_samples = 5*WHISPER_SAMPLE_RATE;

    while (g_running) {
        stream_set_status("waiting for audio ...");

        {
            std::unique_lock<std::mutex> lock(g_mutex);

            if (g_pcmf32.size() < 1024) {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            pcmf32 = std::vector<float>(g_pcmf32.end() - std::min((int64_t) g_pcmf32.size(), window_samples), g_pcmf32.end());
            g_pcmf32.clear();

            // Read dynamic params under the same lock to avoid races with JS setters
            wparams.max_tokens               = g_max_tokens;
            wparams.vad_params.threshold     = g_vad_thold;
            // wparams.vad = true requires a vad_model_path leave disabled until model is bundled
        }

        {
            const auto t_start = std::chrono::high_resolution_clock::now();

            stream_set_status("running whisper ...");

            int ret = whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size());
            if (ret != 0) {
                printf("whisper_full() failed: %d\n", ret);
                break;
            }

            const auto t_end = std::chrono::high_resolution_clock::now();
            printf("stream: whisper_full() returned %d in %f seconds\n", ret, std::chrono::duration<double>(t_end - t_start).count());
        }

        {
            std::string text_heard;
            std::vector<TokenData> tokens_heard;
            float confidence = 0.0f;
            float pt = 0.0f;

            const int n_segments = whisper_full_n_segments(ctx);
            if (n_segments > 0) {
                const int last_segment = n_segments - 1;
                const int n_tokens     = whisper_full_n_tokens(ctx, last_segment);

                if (n_tokens > 0) {
                    // Collect per-token data (text, probability, timestamps).
                    // t0/t1 come from whisper_token_data (populated when token_timestamps=true).
                    // t0/t1 units: centiseconds multiply by 10 for milliseconds.
                    float min_p = 1.0f;
                    for (int i = 0; i < n_tokens; ++i) {
                        const whisper_token_data td = whisper_full_get_token_data(ctx, last_segment, i);
                        const char * token_t = whisper_full_get_token_text(ctx, last_segment, i);

                        tokens_heard.push_back({ token_t ? token_t : "", td.p, td.t0, td.t1 });

                        if (td.p < min_p) min_p = td.p;
                    }

                    // Segment-level confidence: min token probability, clamped then cube-rooted
                    // to spread the dynamic range (raw probs cluster near 0 for low-resource langs)
                    min_p = std::max(0.01f, std::min(0.99f, min_p));
                    confidence = std::pow(min_p, 1.0f / 3.0f);

                    pt = whisper_full_get_token_p(ctx, last_segment, n_tokens - 1);

                    text_heard = whisper_full_get_segment_text(ctx, last_segment);
                    printf("transcribed: %s\n", text_heard.c_str());
                }
            }

            {
                std::lock_guard<std::mutex> lock(g_mutex);
                g_transcribed = std::move(text_heard);
                g_tokens      = std::move(tokens_heard);
                g_confidence  = confidence;
                g_pt          = pt;
            }
        }
    }

    if (index < g_contexts.size()) {
        whisper_free(g_contexts[index]);
        g_contexts[index] = nullptr;
    }
}

EMSCRIPTEN_BINDINGS(stream) {
    emscripten::function("init", emscripten::optional_override([](const std::string & path_model, const std::string & lang) {
        for (size_t i = 0; i < g_contexts.size(); ++i) {
            if (g_contexts[i] == nullptr) {
                g_contexts[i] = whisper_init_from_file_with_params(path_model.c_str(), whisper_context_default_params());
                if (g_contexts[i] != nullptr) {
                    g_running = true;
                    if (g_worker.joinable()) {
                        g_worker.join();
                    }
                    g_worker = std::thread([i, lang]() {
                        stream_main(i, lang);
                    });
                    return i + 1;
                } else {
                    return (size_t) 0;
                }
            }
        }
        return (size_t) 0;
    }));

    emscripten::function("free", emscripten::optional_override([](size_t index) {
        if (g_running) {
            g_running = false;
        }
    }));

    emscripten::function("set_audio", emscripten::optional_override([](size_t index, const emscripten::val & audio) {
        --index;

        if (index >= g_contexts.size()) return -1;
        if (g_contexts[index] == nullptr) return -2;

        {
            std::lock_guard<std::mutex> lock(g_mutex);
            const int n = audio["length"].as<int>();

            emscripten::val heap = emscripten::val::module_property("HEAPU8");
            emscripten::val memory = heap["buffer"];

            g_pcmf32.resize(n);

            emscripten::val memoryView = audio["constructor"].new_(memory, reinterpret_cast<uintptr_t>(g_pcmf32.data()), n);
            memoryView.call<void>("set", audio);
        }

        return 0;
    }));

    emscripten::function("get_transcribed", emscripten::optional_override([]() {
        std::string transcribed;
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            transcribed = std::move(g_transcribed);
        }
        return transcribed;
    }));

    emscripten::function("get_confidence", emscripten::optional_override([]() {
        std::lock_guard<std::mutex> lock(g_mutex);
        return g_confidence;
    }));

    // Returns a JS array of token objects: [{ text, p, t0, t1 }, ...]
    // t0/t1 are in centiseconds (whisper internal unit); multiply by 10 for milliseconds.
    // p < threshold indicates uncertain tokens candidates for reconstruction targeting.
    emscripten::function("get_tokens", emscripten::optional_override([]() {
        std::vector<TokenData> tokens;
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            tokens = g_tokens;
        }

        emscripten::val arr = emscripten::val::array();
        for (size_t i = 0; i < tokens.size(); ++i) {
            emscripten::val obj = emscripten::val::object();
            obj.set("text", tokens[i].text);
            obj.set("p",    tokens[i].p);
            obj.set("t0",   (double) tokens[i].t0);
            obj.set("t1",   (double) tokens[i].t1);
            arr.call<void>("push", obj);
        }
        return arr;
    }));

    // --- Runtime parameter setters ---

    // max_tokens: max output tokens per segment. Default 64.
    // Raise for polysynthetic languages where a single word can span many tokens.
    emscripten::function("set_max_tokens", emscripten::optional_override([](int max_tokens) {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_max_tokens = max_tokens;
    }));

    // vad_thold: speech probability threshold for VAD (maps to wparams.vad_params.threshold).
    // Only takes effect when wparams.vad=true, which requires a VAD model file to be bundled.
    // Kept as a setter so the threshold is ready when VAD support is added.
    emscripten::function("set_vad_thold", emscripten::optional_override([](float thold) {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_vad_thold = thold;
    }));

    emscripten::function("get_status", emscripten::optional_override([]() {
        std::string status;
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            status = g_status_forced.empty() ? g_status : g_status_forced;
        }
        return status;
    }));

    emscripten::function("set_status", emscripten::optional_override([](const std::string & status) {
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            g_status_forced = status;
        }
    }));
}
