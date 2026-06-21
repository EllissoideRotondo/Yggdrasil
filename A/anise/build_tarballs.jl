using BinaryBuilder

name = "anise"
version = v"0.10.2"

sources = [
    GitSource("https://github.com/nyx-space/anise.git",
              "296fdde8d766cb6b544677474d04125e8a9c2d67"),
]

script = raw"""
cd "${WORKSPACE}/srcdir/anise"

mkdir -p anise-c-shim/src anise-c-shim/include

cat > anise-c-shim/Cargo.toml <<'EOF'
[package]
name = "anise-c-shim"
version = "0.10.2"
edition = "2024"
license = "MPL-2.0"

[lib]
name = "anise"
crate-type = ["cdylib", "staticlib"]

[workspace]

[dependencies]
anise = { path = "../anise", default-features = false }
EOF

cat > anise-c-shim/src/lib.rs <<'EOF'
use anise::almanac::Almanac;
use anise::astro::Aberration;
use anise::frames::Frame;
use anise::math::angles::{between_0_360, between_pm_180};
use anise::naif::SPK;
use anise::structure::planetocentric::ellipsoid::Ellipsoid;
use anise::structure::spacecraft::Mass;
use anise::time::Epoch;
use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_double, c_int};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

const ANISE_OK: c_int = 0;
const ANISE_ERROR_NULL_POINTER: c_int = 1;
const ANISE_ERROR_INVALID_UTF8: c_int = 2;
const ANISE_ERROR_RUST: c_int = 3;
const ANISE_ERROR_PANIC: c_int = 4;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("").expect("empty string"));
}

#[repr(C)]
pub struct AniseState {
    pub position_km: [c_double; 3],
    pub velocity_km_s: [c_double; 3],
    pub epoch_et_s: c_double,
    pub frame_ephemeris_id: i32,
    pub frame_orientation_id: i32,
}

pub struct AniseAlmanac {
    inner: Almanac,
}

pub struct AniseSpk {
    inner: SPK,
}

fn set_last_error(message: impl Into<String>) {
    let sanitized = message.into().replace('\0', "\\0");
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = CString::new(sanitized).unwrap_or_else(|_| {
            CString::new("failed to store error message").expect("static string")
        });
    });
}

fn clear_last_error() {
    set_last_error("");
}

fn cstr_to_string(ptr: *const c_char, name: &str) -> Result<String, c_int> {
    if ptr.is_null() {
        set_last_error(format!("{name} is null"));
        return Err(ANISE_ERROR_NULL_POINTER);
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map(|value| value.to_string())
        .map_err(|err| {
            set_last_error(format!("{name} is not valid UTF-8: {err}"));
            ANISE_ERROR_INVALID_UTF8
        })
}

fn run_status(f: impl FnOnce() -> Result<(), String>) -> c_int {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => {
            clear_last_error();
            ANISE_OK
        }
        Ok(Err(err)) => {
            set_last_error(err);
            ANISE_ERROR_RUST
        }
        Err(_) => {
            set_last_error("panic crossing ANISE C ABI");
            ANISE_ERROR_PANIC
        }
    }
}

fn ab_corr_from_name(name: &str) -> Result<Option<Aberration>, String> {
    Aberration::new(name).map_err(|err| err.to_string())
}

#[unsafe(no_mangle)]
pub extern "C" fn anise_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| slot.borrow().as_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn anise_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

#[unsafe(no_mangle)]
pub extern "C" fn anise_between_0_360(angle_deg: c_double) -> c_double {
    between_0_360(angle_deg)
}

#[unsafe(no_mangle)]
pub extern "C" fn anise_between_pm_180(angle_deg: c_double) -> c_double {
    between_pm_180(angle_deg)
}

#[unsafe(no_mangle)]
pub extern "C" fn anise_spacecraft_total_mass_kg(
    dry_mass_kg: c_double,
    prop_mass_kg: c_double,
    extra_mass_kg: c_double,
) -> c_double {
    let mut mass = Mass::from_dry_and_prop_masses(dry_mass_kg, prop_mass_kg);
    mass.extra_mass_kg = extra_mass_kg;
    mass.total_mass_kg()
}

#[unsafe(no_mangle)]
pub extern "C" fn anise_ellipsoid_flattening(
    equatorial_radius_km: c_double,
    polar_radius_km: c_double,
) -> c_double {
    Ellipsoid::from_spheroid(equatorial_radius_km, polar_radius_km).flattening()
}

#[unsafe(no_mangle)]
pub extern "C" fn anise_almanac_new() -> *mut AniseAlmanac {
    clear_last_error();
    Box::into_raw(Box::new(AniseAlmanac {
        inner: Almanac::default(),
    }))
}

#[unsafe(no_mangle)]
pub extern "C" fn anise_almanac_free(almanac: *mut AniseAlmanac) {
    if !almanac.is_null() {
        unsafe {
            drop(Box::from_raw(almanac));
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn anise_almanac_load(
    almanac: *mut *mut AniseAlmanac,
    path: *const c_char,
) -> c_int {
    run_status(|| {
        if almanac.is_null() || unsafe { (*almanac).is_null() } {
            return Err("almanac is null".to_string());
        }
        let path = cstr_to_string(path, "path").map_err(|_| "invalid path".to_string())?;
        let current = unsafe { &**almanac };
        match current.inner.clone().load(&path) {
            Ok(inner) => {
                unsafe {
                    drop(Box::from_raw(*almanac));
                }
                unsafe {
                    *almanac = Box::into_raw(Box::new(AniseAlmanac { inner }));
                }
                Ok(())
            }
            Err(err) => Err(err.to_string()),
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn anise_almanac_num_loaded_spk(
    almanac: *const AniseAlmanac,
    out_count: *mut usize,
) -> c_int {
    run_status(|| {
        if almanac.is_null() || out_count.is_null() {
            return Err("almanac or out_count is null".to_string());
        }
        unsafe {
            *out_count = (*almanac).inner.num_loaded_spk();
        }
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn anise_almanac_spk_domain(
    almanac: *const AniseAlmanac,
    id: i32,
    out_start_et_s: *mut c_double,
    out_end_et_s: *mut c_double,
) -> c_int {
    run_status(|| {
        if almanac.is_null() || out_start_et_s.is_null() || out_end_et_s.is_null() {
            return Err("almanac or output pointer is null".to_string());
        }
        let (start, end) = unsafe { (*almanac).inner.spk_domain(id) }
            .map_err(|err| err.to_string())?;
        unsafe {
            *out_start_et_s = start.to_et_seconds();
            *out_end_et_s = end.to_et_seconds();
        }
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn anise_almanac_spk_ezr(
    almanac: *const AniseAlmanac,
    target: i32,
    epoch_et_s: c_double,
    frame: i32,
    aberration: *const c_char,
    observer: i32,
    out_state: *mut AniseState,
) -> c_int {
    run_status(|| {
        if almanac.is_null() || out_state.is_null() {
            return Err("almanac or out_state is null".to_string());
        }
        let ab_name = cstr_to_string(aberration, "aberration")
            .map_err(|_| "invalid aberration".to_string())?;
        let state = unsafe { &(*almanac).inner }
            .spk_ezr(
                target,
                Epoch::from_et_seconds(epoch_et_s),
                frame,
                observer,
                ab_corr_from_name(&ab_name)?,
            )
            .map_err(|err| err.to_string())?;
        unsafe {
            *out_state = AniseState {
                position_km: [state.radius_km.x, state.radius_km.y, state.radius_km.z],
                velocity_km_s: [
                    state.velocity_km_s.x,
                    state.velocity_km_s.y,
                    state.velocity_km_s.z,
                ],
                epoch_et_s: state.epoch.to_et_seconds(),
                frame_ephemeris_id: state.frame.ephemeris_id,
                frame_orientation_id: state.frame.orientation_id,
            };
        }
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn anise_almanac_state_of(
    almanac: *const AniseAlmanac,
    object: i32,
    observer_ephemeris_id: i32,
    observer_orientation_id: i32,
    epoch_et_s: c_double,
    aberration: *const c_char,
    out_state: *mut AniseState,
) -> c_int {
    run_status(|| {
        if almanac.is_null() || out_state.is_null() {
            return Err("almanac or out_state is null".to_string());
        }
        let ab_name = cstr_to_string(aberration, "aberration")
            .map_err(|_| "invalid aberration".to_string())?;
        let state = unsafe { &(*almanac).inner }
            .state_of(
                object,
                Frame::new(observer_ephemeris_id, observer_orientation_id),
                Epoch::from_et_seconds(epoch_et_s),
                ab_corr_from_name(&ab_name)?,
            )
            .map_err(|err| err.to_string())?;
        unsafe {
            *out_state = AniseState {
                position_km: [state.radius_km.x, state.radius_km.y, state.radius_km.z],
                velocity_km_s: [
                    state.velocity_km_s.x,
                    state.velocity_km_s.y,
                    state.velocity_km_s.z,
                ],
                epoch_et_s: state.epoch.to_et_seconds(),
                frame_ephemeris_id: state.frame.ephemeris_id,
                frame_orientation_id: state.frame.orientation_id,
            };
        }
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn anise_spk_open(path: *const c_char, out_spk: *mut *mut AniseSpk) -> c_int {
    run_status(|| {
        if out_spk.is_null() {
            return Err("out_spk is null".to_string());
        }
        let path = cstr_to_string(path, "path").map_err(|_| "invalid path".to_string())?;
        let spk = SPK::load(&path).map_err(|err| err.to_string())?;
        unsafe {
            *out_spk = Box::into_raw(Box::new(AniseSpk { inner: spk }));
        }
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn anise_spk_free(spk: *mut AniseSpk) {
    if !spk.is_null() {
        unsafe {
            drop(Box::from_raw(spk));
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn anise_spk_write(spk: *const AniseSpk, path: *const c_char) -> c_int {
    run_status(|| {
        if spk.is_null() {
            return Err("spk is null".to_string());
        }
        let path = cstr_to_string(path, "path").map_err(|_| "invalid path".to_string())?;
        unsafe { &(*spk).inner }
            .persist(&path)
            .map_err(|err| err.to_string())
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn anise_spk_summary_count(
    spk: *const AniseSpk,
    out_count: *mut usize,
) -> c_int {
    run_status(|| {
        if spk.is_null() || out_count.is_null() {
            return Err("spk or out_count is null".to_string());
        }
        let summaries = unsafe { &(*spk).inner }
            .data_summaries(None)
            .map_err(|err| err.to_string())?;
        unsafe {
            *out_count = summaries.len();
        }
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn anise_spk_copy(input_path: *const c_char, output_path: *const c_char) -> c_int {
    run_status(|| {
        let input_path = cstr_to_string(input_path, "input_path")
            .map_err(|_| "invalid input_path".to_string())?;
        let output_path = cstr_to_string(output_path, "output_path")
            .map_err(|_| "invalid output_path".to_string())?;
        let spk = SPK::load(&input_path).map_err(|err| err.to_string())?;
        spk.persist(&output_path).map_err(|err| err.to_string())
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn anise_spk_null() -> *mut AniseSpk {
    ptr::null_mut()
}
EOF

cat > anise-c-shim/include/anise.h <<'EOF'
#ifndef ANISE_H
#define ANISE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AniseAlmanac AniseAlmanac;
typedef struct AniseSpk AniseSpk;

typedef struct AniseState {
    double position_km[3];
    double velocity_km_s[3];
    double epoch_et_s;
    int32_t frame_ephemeris_id;
    int32_t frame_orientation_id;
} AniseState;

const char *anise_last_error(void);
const char *anise_version(void);
double anise_between_0_360(double angle_deg);
double anise_between_pm_180(double angle_deg);
double anise_spacecraft_total_mass_kg(double dry_mass_kg, double prop_mass_kg, double extra_mass_kg);
double anise_ellipsoid_flattening(double equatorial_radius_km, double polar_radius_km);

AniseAlmanac *anise_almanac_new(void);
void anise_almanac_free(AniseAlmanac *almanac);
int anise_almanac_load(AniseAlmanac **almanac, const char *path);
int anise_almanac_num_loaded_spk(const AniseAlmanac *almanac, size_t *out_count);
int anise_almanac_spk_domain(const AniseAlmanac *almanac, int32_t id, double *out_start_et_s, double *out_end_et_s);
int anise_almanac_spk_ezr(const AniseAlmanac *almanac, int32_t target, double epoch_et_s, int32_t frame, const char *aberration, int32_t observer, AniseState *out_state);
int anise_almanac_state_of(const AniseAlmanac *almanac, int32_t object, int32_t observer_ephemeris_id, int32_t observer_orientation_id, double epoch_et_s, const char *aberration, AniseState *out_state);

int anise_spk_open(const char *path, AniseSpk **out_spk);
void anise_spk_free(AniseSpk *spk);
int anise_spk_write(const AniseSpk *spk, const char *path);
int anise_spk_summary_count(const AniseSpk *spk, size_t *out_count);
int anise_spk_copy(const char *input_path, const char *output_path);
AniseSpk *anise_spk_null(void);

#ifdef __cplusplus
}
#endif

#endif
EOF

cargo build --release --manifest-path anise-c-shim/Cargo.toml

install -Dvm 755 "anise-c-shim/target/${rust_target}/release/libanise.${dlext}" \
    "${libdir}/libanise.${dlext}"
install -Dvm 644 anise-c-shim/include/anise.h "${includedir}/anise.h"
install_license LICENSE
"""

platforms = supported_platforms()
filter!(p -> arch(p) in ("x86_64", "aarch64"), platforms)
filter!(p -> os(p) in ("linux", "macos"), platforms)
filter!(p -> libc(p) != "musl", platforms)

products = [
    LibraryProduct("libanise", :libanise),
    FileProduct("include/anise.h", :anise_h),
]

dependencies = Dependency[]

build_tarballs(ARGS, name, version, sources, script, platforms, products, dependencies;
               compilers=[:c, :rust], julia_compat="1.10")
