//! The website's tokens (site/src/styles/app.css, desktop/ui/src/styles.css) as Rune colours.
//!
//! Rune blends in linear light, so the site's translucent whites (`rgb(255 255 255 / 0.1)` on
//! `#0a0a0a`) would come out brighter here; they are given as the solid sRGB colours they make on
//! the canvas instead.

use rune::Color;

/// An sRGB hex colour (`0xRRGGBB`) in Rune's linear light.
const fn srgb(hex: u32) -> Color {
    Color {
        r: lin((hex >> 16) & 0xff),
        g: lin((hex >> 8) & 0xff),
        b: lin(hex & 0xff),
        a: 1.0,
    }
}

/// One sRGB channel (0–255) in linear light, by a lookup the compiler can do: `powf` isn't const.
const fn lin(c: u32) -> f32 {
    LINEAR[c as usize]
}

const LINEAR: [f32; 256] = {
    let mut t = [0.0f32; 256];
    let mut i = 0;
    while i < 256 {
        let c = i as f64 / 255.0;
        t[i] = if c <= 0.04045 {
            (c / 12.92) as f32
        } else {
            // ((c + 0.055) / 1.055)^2.4 as exp(2.4 · ln x), with a series ln and exp (const fn).
            let x = (c + 0.055) / 1.055;
            exp(2.4 * ln(x)) as f32
        };
        i += 1;
    }
    t
};

/// ln x for x in (0, 1], by atanh's series: ln x = 2·atanh((x−1)/(x+1)).
const fn ln(x: f64) -> f64 {
    let y = (x - 1.0) / (x + 1.0);
    let y2 = y * y;
    let mut term = y;
    let mut sum = 0.0;
    let mut n = 1.0;
    let mut k = 0;
    while k < 60 {
        sum += term / n;
        term *= y2;
        n += 2.0;
        k += 1;
    }
    2.0 * sum
}

/// e^x for x ≤ 0, by its Taylor series.
const fn exp(x: f64) -> f64 {
    let mut term = 1.0;
    let mut sum = 1.0;
    let mut k = 1;
    while k < 60 {
        term *= x / k as f64;
        sum += term;
        k += 1;
    }
    sum
}

pub const CANVAS: Color = srgb(0x0a0a0a);
/// `--color-panel` on the canvas.
pub const PANEL: Color = srgb(0x0e0e0e);
/// `--color-line`.
pub const LINE: Color = srgb(0x232323);
/// `--color-fill`.
pub const FILL: Color = srgb(0x191919);
/// `--color-fill-strong`.
pub const FILL_STRONG: Color = srgb(0x232323);
pub const FG: Color = srgb(0xfafafa);
/// `--color-body`: running text and figures' captions.
pub const BODY: Color = srgb(0xc4c4c4);
pub const MUTED: Color = srgb(0x8a8a8a);
pub const FAINT: Color = srgb(0x5c5c5c);
/// `--color-code` on a panel: the curl example's band.
pub const CODE: Color = srgb(0x161616);
pub const BTN: Color = srgb(0xfafafa);
pub const BTN_HOVER: Color = srgb(0xffffff);
pub const BTN_FG: Color = srgb(0x0a0a0a);
pub const OK: Color = srgb(0x8fd3ae);
/// `--color-cap-bg`, the green at 14 % on the canvas: a capability's tag.
pub const CAP_BG: Color = srgb(0x1d2621);
pub const BAD: Color = srgb(0xf85149);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_matches_the_standard_curve() {
        for c in [0u32, 10, 64, 128, 200, 255] {
            let x = c as f64 / 255.0;
            let want = if x <= 0.04045 {
                x / 12.92
            } else {
                ((x + 0.055) / 1.055).powf(2.4)
            };
            assert!(
                (lin(c) as f64 - want).abs() < 1e-6,
                "{c}: {} vs {want}",
                lin(c)
            );
        }
    }
}
