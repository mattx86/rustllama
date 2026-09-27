//! Functional check for the Level Zero Sysman energy-counter math
//! the power-aware tuning loop relies on.
//!
//! Run:
//! ```
//! cargo run --release -p rustllama-engine --example power_aware_verify
//! ```
//!
//! What it does:
//!
//! 1. Opens the Sysman probe on device 0.
//! 2. Samples the cumulative energy counter at t=0.
//! 3. Burns CPU + GPU briefly (the workload doesn't matter — we
//!    just need wall time to elapse so the counter advances).
//! 4. Samples again at t=1s.
//! 5. Computes `watts = ΔE_µJ / Δt_µs` and prints both the raw
//!    `(e0, t0, e1, t1)` quadruple AND the derived watts.
//!
//! Expected output on Iris Xe (integrated graphics, no discrete
//! power probe): a non-zero `ΔE` and a `watts` value in the 0.5-15W
//! range. If `ΔE = 0` or `watts < 0.1`, the Sysman power domain
//! isn't reporting — fall back to the placement sweep without
//! `--optimize-for=watts`.
//!
//! Requires `ZES_ENABLE_SYSMAN=1` in the environment (Intel driver
//! contract). Without it, `Sysman::load` returns
//! `L0Error::Uninitialized` and the probe never opens.

use rustllama_l0_sys::{LevelZero, Sysman};
use std::time::{Duration, Instant};

fn main() {
    // Required for Intel L0 Sysman to expose any data.
    std::env::set_var("ZES_ENABLE_SYSMAN", "1");

    let sysman = match Sysman::load() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Sysman load failed: {e:?}");
            eprintln!(
                "  expected on non-Intel hosts or when oneAPI's L0 loader isn't installed"
            );
            return;
        }
    };
    let l0 = match LevelZero::load() {
        Ok(l) => l,
        Err(e) => {
            eprintln!("LevelZero load failed: {e:?}");
            return;
        }
    };
    let drivers = match l0.drivers() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("driver enumeration failed: {e:?}");
            return;
        }
    };
    if drivers.is_empty() {
        eprintln!("no L0 drivers visible");
        return;
    }
    let devices = match l0.devices(drivers[0]) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("device enumeration failed: {e:?}");
            return;
        }
    };
    if devices.is_empty() {
        eprintln!("no L0 devices visible");
        return;
    }
    let device = devices[0];
    eprintln!("Sysman + device probe ready (device 0)");
    eprintln!();

    // Sample energy at t=0.
    let r0 = sysman.probe(device);
    let e0 = match r0.energy_counter {
        Some(c) => c,
        None => {
            eprintln!("Sysman reports no power-domain on device 0");
            eprintln!("  expected on Iris Xe — integrated graphics often hide the energy register");
            eprintln!("  power-aware tuning will silently fall back to throughput-only");
            return;
        }
    };
    eprintln!("t=0:  energy = {:>14} µJ, timestamp = {:>12} µs", e0.energy_uj, e0.timestamp_us);

    // Burn ~1 second. CPU spinning to make sure SOMETHING draws
    // power; the GPU may or may not be active depending on driver
    // power state. The energy delta should still register because
    // Sysman tracks package power, not just compute-unit power.
    let burn_start = Instant::now();
    let mut acc: f64 = 0.0;
    while burn_start.elapsed() < Duration::from_secs(1) {
        // Compute something the compiler can't hoist away.
        for i in 0..10_000 {
            acc += (i as f64).sqrt().sin();
        }
    }
    // Touch acc so it doesn't get DCEd.
    if acc.is_nan() {
        eprintln!("nan: {acc}");
    }

    let r1 = sysman.probe(device);
    let e1 = match r1.energy_counter {
        Some(c) => c,
        None => {
            eprintln!("second sample missing energy_counter — driver hiccup");
            return;
        }
    };
    eprintln!("t=1s: energy = {:>14} µJ, timestamp = {:>12} µs", e1.energy_uj, e1.timestamp_us);

    let de_uj = e1.energy_uj.saturating_sub(e0.energy_uj) as f64;
    let dt_us = e1.timestamp_us.saturating_sub(e0.timestamp_us) as f64;
    eprintln!();
    eprintln!("ΔE = {de_uj:.0} µJ");
    eprintln!("Δt = {dt_us:.0} µs");
    if dt_us > 0.0 {
        let watts = de_uj / dt_us; // µJ/µs == W
        eprintln!("derived watts = {watts:.3} W");
        if watts < 0.1 {
            eprintln!();
            eprintln!("WARN: derived watts < 0.1W — Sysman is reading but the");
            eprintln!("counter isn't advancing meaningfully. Power-aware tuning");
            eprintln!("will produce useless tok/joule numbers on this device.");
        } else if watts > 200.0 {
            eprintln!();
            eprintln!("WARN: derived watts > 200W — counter overflow or timestamp");
            eprintln!("wraparound. Double-check the L0 Sysman implementation.");
        } else {
            eprintln!();
            eprintln!("OK: power-aware tuning's energy-delta math works on this device.");
        }
    } else {
        eprintln!("Δt = 0 — timestamps didn't advance; Sysman is broken");
    }
}
