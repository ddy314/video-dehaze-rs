use crate::dcp::RgbImageF32;

pub fn psnr(reference: &RgbImageF32, candidate: &RgbImageF32) -> Option<f32> {
    if reference.width != candidate.width || reference.height != candidate.height {
        return None;
    }

    let mse = reference
        .data
        .iter()
        .zip(candidate.data.iter())
        .flat_map(|(a, b)| {
            [
                (a[0] - b[0]).powi(2),
                (a[1] - b[1]).powi(2),
                (a[2] - b[2]).powi(2),
            ]
        })
        .sum::<f32>()
        / (reference.data.len() * 3).max(1) as f32;

    if mse <= f32::EPSILON {
        Some(f32::INFINITY)
    } else {
        Some(10.0 * (1.0 / mse).log10())
    }
}

pub fn ssim(reference: &RgbImageF32, candidate: &RgbImageF32) -> Option<f32> {
    if reference.width != candidate.width || reference.height != candidate.height {
        return None;
    }

    let a = reference.luma();
    let b = candidate.luma();
    let n = a.len().max(1) as f32;
    let mean_a = a.iter().sum::<f32>() / n;
    let mean_b = b.iter().sum::<f32>() / n;
    let var_a = a.iter().map(|v| (v - mean_a).powi(2)).sum::<f32>() / (n - 1.0).max(1.0);
    let var_b = b.iter().map(|v| (v - mean_b).powi(2)).sum::<f32>() / (n - 1.0).max(1.0);
    let cov = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - mean_a) * (y - mean_b))
        .sum::<f32>()
        / (n - 1.0).max(1.0);
    let c1 = 0.01f32.powi(2);
    let c2 = 0.03f32.powi(2);

    Some(
        ((2.0 * mean_a * mean_b + c1) * (2.0 * cov + c2))
            / ((mean_a.powi(2) + mean_b.powi(2) + c1) * (var_a + var_b + c2)),
    )
}

pub fn color_delta(reference: &RgbImageF32, candidate: &RgbImageF32) -> Option<f32> {
    if reference.width != candidate.width || reference.height != candidate.height {
        return None;
    }
    let total = reference
        .data
        .iter()
        .zip(candidate.data.iter())
        .map(|(a, b)| ((a[0] - b[0]).abs() + (a[1] - b[1]).abs() + (a[2] - b[2]).abs()) / 3.0)
        .sum::<f32>();
    Some(total / reference.data.len().max(1) as f32)
}

pub fn flicker(previous: &RgbImageF32, current: &RgbImageF32) -> Option<f32> {
    if previous.width != current.width || previous.height != current.height {
        return None;
    }
    let a = previous.luma();
    let b = current.luma();
    let total = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f32>();
    Some(total / a.len().max(1) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_handle_identical_images() {
        let img = RgbImageF32::blank(2, 2, [0.5, 0.5, 0.5]);
        assert!(psnr(&img, &img).unwrap().is_infinite());
        assert!(ssim(&img, &img).unwrap() > 0.99);
        assert_eq!(color_delta(&img, &img).unwrap(), 0.0);
        assert_eq!(flicker(&img, &img).unwrap(), 0.0);
    }
}
