//! Smoke test for `rustllama tuning show`. The command formats data
//! the integration tests in `tuner_cache_persistence.rs` already
//! verify on the persistence side; this test just confirms the
//! human-readable path runs to completion under the two paths most
//! users hit:
//!
//!   - default config + no SYCL device (mock-mode test host) →
//!     prints the "no SYCL device visible" hint, returns Ok
//!   - a populated cache reachable via the default-cache-dir path
//!     would print the placement / batch_size rows, but the
//!     integration test can't seed the default-cache-dir without
//!     racing other tests — that path is exercised via the same
//!     `load_cache` round-trip the persister tests already cover.

use rustllama_cli::cmd_tuning_show;

#[test]
fn tuning_show_runs_to_completion_with_default_config() {
    // Use a tempfile config — the command reads `auto_apply_*` from
    // it, and we want the test to be insensitive to whatever lives
    // in the user's real config.
    let cfg = std::env::temp_dir().join("rustllama-tuning-show-default.toml");
    std::fs::write(
        &cfg,
        r#"
[server]
port = 11434
"#,
    )
    .expect("write config");

    let result = cmd_tuning_show(&cfg);
    assert!(
        result.is_ok(),
        "tuning show must not error on default config: {:?}",
        result
    );

    let _ = std::fs::remove_file(&cfg);
}

#[test]
fn tuning_show_runs_to_completion_with_explicit_auto_apply_off() {
    let cfg = std::env::temp_dir().join("rustllama-tuning-show-no-apply.toml");
    std::fs::write(
        &cfg,
        r#"
[server]
port = 11434

[tuning]
auto_apply_placement = false
auto_apply_batch_size = false
"#,
    )
    .expect("write config");

    let result = cmd_tuning_show(&cfg);
    assert!(
        result.is_ok(),
        "tuning show must not error when auto_apply_* are false: {:?}",
        result
    );

    let _ = std::fs::remove_file(&cfg);
}

#[test]
fn tuning_show_tolerates_missing_config_file() {
    // The command falls back to Config::default() when load fails,
    // so a path that doesn't exist still works — the user might
    // run `tuning show` before they've ever saved a config.
    let cfg = std::env::temp_dir().join("rustllama-tuning-show-missing-config.toml");
    let _ = std::fs::remove_file(&cfg);
    assert!(!cfg.exists(), "test fixture: config file must not exist");

    let result = cmd_tuning_show(&cfg);
    assert!(
        result.is_ok(),
        "tuning show must tolerate a missing config file (falls back to default): {:?}",
        result
    );
}
