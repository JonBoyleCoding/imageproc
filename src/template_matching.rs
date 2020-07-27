//! Functions for performing template matching.
use crate::definitions::Image;
use crate::integral_image::{integral_squared_image, sum_image_pixels, ArrayData};
use crate::map::{map_pixels, WithChannel};
use crate::rect::Rect;
use image::{GenericImageView, Luma, Pixel, Primitive};
use num::traits::NumAssign;
use num::{NumCast, ToPrimitive};

/// Method used to compute the matching score between a template and an image region.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MatchTemplateMethod {
    /// Sum of the squares of the difference between image and template pixel
    /// intensities.
    ///
    /// Smaller values are better.
    SumOfSquaredErrors,
    /// Divides the sum computed using `SumOfSquaredErrors` by a normalization term.
    SumOfSquaredErrorsNormalized,
    /// Cross Correlation
    ///
    /// Higher values are better.
    CrossCorrelation,
    /// Divides the sum computed using `CrossCorrelation` by a normalization term.
    CrossCorrelationNormalized,
    /// Correlation Coefficient
    ///
    /// -1 is negative correlation
    /// 0 is no correlation
    /// +1 is perfect correlation
    CorrelationCoefficient,
    /// Divides the sum computed using `CorrelationCoefficient` by a normalization term.
    CorrelationCoefficientNormalized,
}

/// Slides a `template` over an `image` and scores the match at each point using
/// the requested `method`.
///
/// The returned image has dimensions `image.width() - template.width() + 1` by
/// `image.height() - template.height() + 1`.
///
/// # Panics
///
/// If either dimension of `template` is not strictly less than the corresponding dimension
/// of `image`, or Correlation Coefficient being calculated with more than one channel per
/// pixel.
pub fn match_template<P>(
    image: &Image<P>,
    template: &Image<P>,
    method: MatchTemplateMethod,
) -> Image<Luma<f32>>
where
    P: Pixel + 'static + WithChannel<f32> + ArrayData,
    P::Subpixel: NumAssign + NumCast + 'static,
    <P as WithChannel<f32>>::Pixel: ArrayData,
{
    let (image_width, image_height) = image.dimensions();
    let (template_width, template_height) = template.dimensions();

    assert!(
        image_width >= template_width,
        "image width must be greater than or equal to template width"
    );
    assert!(
        image_height >= template_height,
        "image height must be greater than or equal to template height"
    );

    use MatchTemplateMethod::*;

    match method {
        CorrelationCoefficient => {
            return match_template_correlation_coefficient(image, template, false);
        }
        CorrelationCoefficientNormalized => {
            return match_template_correlation_coefficient(image, template, true);
        }
        SumOfSquaredErrors
        | SumOfSquaredErrorsNormalized
        | CrossCorrelation
        | CrossCorrelationNormalized => {
            let should_normalize = match method {
                MatchTemplateMethod::SumOfSquaredErrorsNormalized
                | MatchTemplateMethod::CrossCorrelationNormalized => true,
                _ => false,
            };

            let image_squared_integral = if should_normalize {
                Some(integral_squared_image::<_, f32>(image))
            } else {
                None
            };

            let template_squared_sum = if should_normalize {
                Some(sum_squares(template))
            } else {
                None
            };

            let mut result = Image::new(
                image_width - template_width + 1,
                image_height - template_height + 1,
            );

            for y in 0..result.height() {
                for x in 0..result.width() {
                    let mut score = 0f32;

                    for dy in 0..template_height {
                        for dx in 0..template_width {
                            let image_pixel = unsafe { image.unsafe_get_pixel(x + dx, y + dy) };
                            let template_pixel = unsafe { template.unsafe_get_pixel(dx, dy) };

                            for c in 0..P::CHANNEL_COUNT {
                                let image_value =
                                    image_pixel.channels()[c as usize].to_f32().unwrap();
                                let template_value =
                                    template_pixel.channels()[c as usize].to_f32().unwrap();

                                score += match method {
                                    SumOfSquaredErrors | SumOfSquaredErrorsNormalized => {
                                        (image_value - template_value).powf(2.0)
                                    }
                                    CrossCorrelation | CrossCorrelationNormalized => {
                                        image_value * template_value
                                    }
                                    _ => {
                                        panic!("Should not be possible to reach this line of code.")
                                    }
                                };
                            }
                        }
                    }

                    if let (&Some(ref i), &Some(t)) =
                        (&image_squared_integral, &template_squared_sum)
                    {
                        let region =
                            Rect::at(x as i32, y as i32).of_size(template_width, template_height);
                        let norm = normalization_term(i, t, region);
                        if norm > 0.0 {
                            score /= norm;
                        }
                    }

                    result.put_pixel(x, y, Luma([score]));
                }
            }

            return result;
        }
    };
}

/// Performs template match for correlation coefficient
fn match_template_correlation_coefficient<P>(
    image: &Image<P>,
    template: &Image<P>,
    normalize: bool,
) -> Image<Luma<f32>>
where
    P: Pixel + 'static,
    P::Subpixel: NumCast + NumAssign + 'static,
{
    let (image_width, image_height) = image.dimensions();
    let (template_width, template_height) = template.dimensions();

    debug_assert!(
        image_width >= template_width,
        "image width must be greater than or equal to template width"
    );
    debug_assert!(
        image_height >= template_height,
        "image height must be greater than or equal to template height"
    );

    assert_eq!(
        P::CHANNEL_COUNT,
        1,
        "image must have only one channel for correlation coeffficient"
    );

    // Create Result Image
    let mut result = Image::new(
        image_width - template_width + 1,
        image_height - template_height + 1,
    );

    // Pre-calculate mean / sample stddev of template
    let template_mean = get_mean(
        template,
        Rect::at(0, 0).of_size(template_width, template_height),
    );

    // Pre-calculate top of variance equation
    let variance_tops: Image<Luma<f32>> = map_pixels(template, |_x, _y, p| {
        let tp_value: f32 = NumCast::from(p.channels()[0]).unwrap();
        Luma([(tp_value - template_mean)])
    });

    let variance_tops_squared: Image<Luma<f32>> = map_pixels(&variance_tops, |_x, _y, pxl| {
        let tp_value: f32 = pxl.0[0];
        Luma([tp_value.powi(2)])
    });

    // Calculate the Correlation Coefficient
    for y in 0..result.height() {
        for x in 0..result.width() {
            // Calculate mean / sample stddev of window
            let image_mean = get_mean(
                image,
                Rect::at(x as i32, y as i32).of_size(template_width, template_height),
            );

            let mut top_sum: f32 = 0f32;
            let mut bottom_x: f32 = 0f32;
            let mut bottom_y: f32 = 0f32;

            for dy in 0..template_height {
                for dx in 0..template_width {
                    let image_pixel = unsafe { image.unsafe_get_pixel(x + dx, y + dy) };
                    let template_var_pixel = unsafe { variance_tops.unsafe_get_pixel(dx, dy) };

                    let im_value: f32 = NumCast::from(image_pixel.channels()[0]).unwrap();
                    let tp_var_value: f32 = template_var_pixel.0[0];

                    // Sum the Multiply the variance tops of image and template
                    let im_var_top = im_value - image_mean;
                    top_sum += im_var_top * tp_var_value;

                    if normalize {
                        let template_var_sq_pixel =
                            unsafe { variance_tops_squared.unsafe_get_pixel(dx, dy) };

                        bottom_x += im_var_top.powi(2);
                        bottom_y += template_var_sq_pixel.0[0];
                    }
                }
            }

            if normalize {
                result.put_pixel(x, y, Luma([top_sum / (bottom_x.sqrt() * bottom_y.sqrt())]));
            } else {
                result.put_pixel(x, y, Luma([top_sum]));
            }
        }
    }

    result
}

/// Returns Mean and Sample Standard Deviation of region within image
///
/// If multi-channel image passed through, will only perform on first channel.
///
/// Return values are (mean, sample_standard_deviation)
fn get_mean<P>(image: &Image<P>, region: Rect) -> f32
where
    P: Pixel + 'static,
    P::Subpixel: NumAssign + 'static,
{
    debug_assert!(region.left() >= 0);
    debug_assert!(region.top() >= 0);
    debug_assert!(region.right() < image.width() as i32);
    debug_assert!(region.bottom() < image.height() as i32);

    // Number of pixels within window
    let n = (region.width() * region.height()) as f32;

    // Calculate Total Sum
    let mut sum_template: f32 = 0f32;

    for dy in region.top()..region.bottom() + 1 {
        for dx in region.left()..region.right() + 1 {
            let template_pixel = unsafe { image.unsafe_get_pixel(dx as u32, dy as u32) };

            let tp_value: f32 = NumCast::from(template_pixel.channels()[0]).unwrap();
            sum_template += tp_value;
        }
    }

    // Calculate and return mean
    return sum_template / n;
}

fn sum_squares<P>(template: &Image<P>) -> f32
where
    P: Pixel + 'static,
    P::Subpixel: NumAssign + 'static,
{
    template
        .pixels()
        .map(|p| {
            p.channels()
                .iter()
                .map(|pv| pv.to_f32().unwrap().powf(2.0))
                .sum::<f32>()
        })
        .sum()
}

/// Returns the square root of the product of the sum of squares of
/// pixel intensities in template and the provided region of image.
fn normalization_term<P>(
    image_squared_integral: &Image<P>,
    template_squared_sum: f32,
    region: Rect,
) -> f32
where
    P: Pixel + 'static + ArrayData,
    P::Subpixel: NumAssign + 'static,
{
    let image_sum = sum_image_pixels(
        image_squared_integral,
        region.left() as u32,
        region.top() as u32,
        region.right() as u32,
        region.bottom() as u32,
    );

    let image_sum = image_sum.iter().map(|v| v.to_f32().unwrap()).sum::<f32>();

    (image_sum * template_squared_sum).sqrt()
}

/// The largest and smallest values in an image,
/// together with their locations.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Extremes<T> {
    /// The largest value in an image.
    pub max_value: T,
    /// The smallest value in an image.
    pub min_value: T,
    /// The coordinates of the largest value in an image.
    pub max_value_location: (u32, u32),
    /// The coordinates of the smallest value in an image.
    pub min_value_location: (u32, u32),
}

/// Finds the largest and smallest values in an image and their locations.
/// If there are multiple such values then the lexicographically smallest is returned.
pub fn find_extremes<T>(image: &Image<Luma<T>>) -> Extremes<T>
where
    T: Primitive + 'static,
{
    assert!(
        image.width() > 0 && image.height() > 0,
        "image must be non-empty"
    );

    let mut min_value = image.get_pixel(0, 0)[0];
    let mut max_value = image.get_pixel(0, 0)[0];

    let mut min_value_location = (0, 0);
    let mut max_value_location = (0, 0);

    for (x, y, p) in image.enumerate_pixels() {
        if p[0] < min_value {
            min_value = p[0];
            min_value_location = (x, y);
        }
        if p[0] > max_value {
            max_value = p[0];
            max_value_location = (x, y);
        }
    }

    Extremes {
        max_value,
        min_value,
        max_value_location,
        min_value_location,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::gray_bench_image;
    use image::GrayImage;
    use test::{black_box, Bencher};

    #[test]
    #[should_panic]
    fn match_template_panics_if_image_width_does_is_less_than_template_width() {
        let _ = match_template(
            &GrayImage::new(5, 5),
            &GrayImage::new(6, 5),
            MatchTemplateMethod::SumOfSquaredErrors,
        );
    }

    #[test]
    #[should_panic]
    fn match_template_panics_if_image_height_is_less_than_template_height() {
        let _ = match_template(
            &GrayImage::new(5, 5),
            &GrayImage::new(5, 6),
            MatchTemplateMethod::SumOfSquaredErrors,
        );
    }

    #[test]
    fn match_template_handles_template_of_same_size_as_image() {
        assert_pixels_eq!(
            match_template(
                &GrayImage::new(5, 5),
                &GrayImage::new(5, 5),
                MatchTemplateMethod::SumOfSquaredErrors
            ),
            gray_image!(type: f32, 0.0)
        );
    }

    #[test]
    fn match_template_normalization_handles_zero_norm() {
        assert_pixels_eq!(
            match_template(
                &GrayImage::new(1, 1),
                &GrayImage::new(1, 1),
                MatchTemplateMethod::SumOfSquaredErrorsNormalized
            ),
            gray_image!(type: f32, 0.0)
        );
    }

    #[test]
    fn match_template_sum_of_squared_errors() {
        let image = gray_image!(
            1, 4, 2;
            2, 1, 3;
            3, 3, 4
        );
        let template = gray_image!(
            1, 2;
            3, 4
        );

        let actual = match_template(&image, &template, MatchTemplateMethod::SumOfSquaredErrors);
        let expected = gray_image!(type: f32,
            14.0, 14.0;
            3.0, 1.0
        );

        assert_pixels_eq!(actual, expected);
    }

    #[test]
    fn match_template_rgb_sum_of_squared_errors() {
        let image = rgb_image!(
            [1,  2,  3], [ 4,  5,  6], [7, 8, 9];
            [10,  11,  12], [13, 14, 15], [16, 17, 18];
            [1,  2,  3], [ 4,  5,  6], [7, 8, 9]);

        let template = rgb_image!(
            [1, 2, 3];
            [11, 12, 13]
        );

        let actual = match_template(&image, &template, MatchTemplateMethod::SumOfSquaredErrors);
        let expected = gray_image!(type: f32,
            3.0, 39.0, 183.0;
            543.0, 579.0, 723.0
        );

        assert_pixels_eq!(actual, expected);
    }

    #[test]
    fn match_template_sum_of_squared_errors_normalized() {
        let image = gray_image!(
            1, 4, 2;
            2, 1, 3;
            3, 3, 4
        );
        let template = gray_image!(
            1, 2;
            3, 4
        );

        let actual = match_template(
            &image,
            &template,
            MatchTemplateMethod::SumOfSquaredErrorsNormalized,
        );
        let tss = 30f32;
        let expected = gray_image!(type: f32,
            14.0 / (22.0 * tss).sqrt(), 14.0 / (30.0 * tss).sqrt();
            3.0 / (23.0 * tss).sqrt(), 1.0 / (35.0 * tss).sqrt()
        );

        assert_pixels_eq!(actual, expected);
    }

    #[test]
    fn match_template_rgb_sum_of_squared_errors_normalized() {
        let image = rgb_image!(
            [1,  2,  3], [ 4,  5,  6], [7, 8, 9];
            [10,  11,  12], [13, 14, 15], [16, 17, 18]);

        let template = rgb_image!(
            [1, 2, 3];
            [11, 12, 13]
        );

        let actual = match_template(
            &image,
            &template,
            MatchTemplateMethod::SumOfSquaredErrorsNormalized,
        );
        // Not yet correct expected
        let expected = gray_image!(type: f32,
            0.007280524, 0.07134486, 0.26518285
        );

        assert_pixels_eq!(actual, expected);
    }

    #[test]
    fn match_template_cross_correlation() {
        let image = gray_image!(
            1, 4, 2;
            2, 1, 3;
            3, 3, 4
        );
        let template = gray_image!(
            1, 2;
            3, 4
        );

        let actual = match_template(&image, &template, MatchTemplateMethod::CrossCorrelation);
        let expected = gray_image!(type: f32,
            19.0, 23.0;
            25.0, 32.0
        );

        assert_pixels_eq!(actual, expected);
    }

    #[test]
    fn match_template_rgb_cross_correlation() {
        let image = rgb_image!(
            [1,  2,  3], [ 4,  5,  6], [7, 8, 9];
            [10,  11,  12], [13, 14, 15], [16, 17, 18];
            [1,  2,  3], [ 4,  5,  6], [7, 8, 9]);

        let template = rgb_image!(
            [1, 2, 3];
            [11, 12, 13]
        );

        let actual = match_template(&image, &template, MatchTemplateMethod::CrossCorrelation);
        let expected = gray_image!(type: f32,
            412.0, 538.0, 664.0;
            142.0, 268.0, 394.0
        );

        assert_pixels_eq!(actual, expected);
    }

    #[test]
    fn match_template_cross_correlation_normalized() {
        let image = gray_image!(
            1, 4, 2;
            2, 1, 3;
            3, 3, 4
        );
        let template = gray_image!(
            1, 2;
            3, 4
        );

        let actual = match_template(
            &image,
            &template,
            MatchTemplateMethod::CrossCorrelationNormalized,
        );
        let tss = 30f32;
        let expected = gray_image!(type: f32,
            19.0 / (22.0 * tss).sqrt(), 23.0 / (30.0 * tss).sqrt();
            25.0 / (23.0 * tss).sqrt(), 32.0 / (35.0 * tss).sqrt()
        );

        assert_pixels_eq!(actual, expected);
    }

    #[test]
    fn match_template_rgb_cross_correlation_normalised() {
        let image = rgb_image!(
            [1,  2,  3], [ 4,  5,  6], [7, 8, 9];
            [10,  11,  12], [13, 14, 15], [16, 17, 18];
            [1,  2,  3], [ 4,  5,  6], [7, 8, 9]);

        let template = rgb_image!(
            [1, 2, 3];
            [11, 12, 13]
        );

        let actual = match_template(
            &image,
            &template,
            MatchTemplateMethod::CrossCorrelationNormalized,
        );
        let expected = gray_image!(type: f32,
            0.9998586, 0.9841932, 0.96219355;
            0.34461147, 0.49026725, 0.57094014
        );

        assert_pixels_eq!(actual, expected);
    }

    #[test]
    fn match_template_correlation_coefficient() {
        let image = gray_image!(
            1, 4, 2;
            2, 1, 3;
            3, 3, 4
        );
        let template = gray_image!(
            1, 2;
            3, 4
        );

        let actual = match_template(
            &image,
            &template,
            MatchTemplateMethod::CorrelationCoefficient,
        );

        // Expected results from OpenCV's implementation
        let expected = gray_image!(type: f32,
            -1., -2.;
            2.5, 4.5
        );

        assert_pixels_eq!(actual, expected);
    }

    #[test]
    fn match_template_correlation_coefficient_normalized() {
        let image = gray_image!(
            1, 4, 2;
            2, 1, 3;
            3, 3, 4
        );
        let template = gray_image!(
            1, 2;
            3, 4
        );

        let actual = match_template(
            &image,
            &template,
            MatchTemplateMethod::CorrelationCoefficientNormalized,
        );

        // Expected results from OpenCV's implementation
        let expected = gray_image!(type: f32,
            -0.18257418, -0.4;
            0.6741998, 0.9233805
        );

        assert_pixels_eq!(actual, expected);
    }

    #[test]
    fn match_template_mean() {
        let image = gray_image!(
            1, 4, 2;
            2, 1, 3;
            3, 3, 4
        );
        let actual = get_mean(&image, Rect::at(0, 0).of_size(2, 2));
        let expected = 2f32;

        assert_eq!(actual, expected);
    }

    macro_rules! bench_match_template {
        ($name:ident, image_size: $s:expr, template_size: $t:expr, method: $m:expr) => {
            #[bench]
            fn $name(b: &mut Bencher) {
                let image = gray_bench_image($s, $s);
                let template = gray_bench_image($t, $t);
                b.iter(|| {
                    let result =
                        match_template(&image, &template, MatchTemplateMethod::SumOfSquaredErrors);
                    black_box(result);
                })
            }
        };
    }

    bench_match_template!(
        bench_match_template_s100_t1_sse,
        image_size: 100,
        template_size: 1,
        method: MatchTemplateMethod::SumOfSquaredErrors);

    bench_match_template!(
        bench_match_template_s100_t4_sse,
        image_size: 100,
        template_size: 4,
        method: MatchTemplateMethod::SumOfSquaredErrors);

    bench_match_template!(
        bench_match_template_s100_t16_sse,
        image_size: 100,
        template_size: 16,
        method: MatchTemplateMethod::SumOfSquaredErrors);

    bench_match_template!(
        bench_match_template_s100_t1_sse_norm,
        image_size: 100,
        template_size: 1,
        method: MatchTemplateMethod::SumOfSquaredErrorsNormalized);

    bench_match_template!(
        bench_match_template_s100_t4_sse_norm,
        image_size: 100,
        template_size: 4,
        method: MatchTemplateMethod::SumOfSquaredErrorsNormalized);

    bench_match_template!(
        bench_match_template_s100_t16_sse_norm,
        image_size: 100,
        template_size: 16,
        method: MatchTemplateMethod::SumOfSquaredErrorsNormalized);

    #[test]
    fn test_find_extremes() {
        let image = gray_image!(
            10,  7,  8,  1;
             9, 15,  4,  2
        );

        let expected = Extremes {
            max_value: 15,
            min_value: 1,
            max_value_location: (1, 1),
            min_value_location: (3, 0),
        };

        assert_eq!(find_extremes(&image), expected);
    }
}
