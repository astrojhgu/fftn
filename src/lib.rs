#![allow(dead_code)]
extern crate rustfft;
extern crate ndarray;

pub use rustfft::{FFTnum};
pub use rustfft::num_complex::Complex;
use rustfft::{FFTplanner};
use rustfft::num_traits::{Zero};
use ndarray::{ArrayViewMut, ArrayViewMut2, Dimension};


pub fn fft2<T: FFTnum+From<u32>>(input: &mut ArrayViewMut2<Complex<T>>, output: &mut ArrayViewMut2<Complex<T>>) {
    fftnd(input, output, &[0,1]);
}

pub fn ifft2<T:FFTnum+From<u32>>(input: &mut ArrayViewMut2<Complex<T>>, output: &mut ArrayViewMut2<Complex<T>>) 
{
    ifftnd(input, output, &[1,0]);
}

pub fn fftn<T: FFTnum+From<u32>, D: Dimension>(input: &mut ArrayViewMut<Complex<T>, D>, output: &mut ArrayViewMut<Complex<T>, D>, axis: usize) {
    _fftn(input, output, axis, false);
}

pub fn ifftn<T: FFTnum+From<u32>, D: Dimension>(input: &mut ArrayViewMut<Complex<T>, D>, output: &mut ArrayViewMut<Complex<T>, D>, axis: usize) {
    _fftn(input, output, axis, true);
}

pub fn fftnd<T: FFTnum+From<u32>, D: Dimension>(input: &mut ArrayViewMut<Complex<T>, D>, output: &mut ArrayViewMut<Complex<T>, D>, axes: &[usize]) {
    _fftnd(input, output, axes, false);
}

pub fn ifftnd<T: FFTnum+From<u32>, D: Dimension>(input: &mut ArrayViewMut<Complex<T>, D>, output: &mut ArrayViewMut<Complex<T>, D>, axes: &[usize]) {
    _fftnd(input, output, axes, true);
}



fn _fftn<T: FFTnum+From<u32>, D: Dimension>(input: &mut ArrayViewMut<Complex<T>, D>, output: &mut ArrayViewMut<Complex<T>, D>, axis: usize, inverse: bool) {
    if inverse {
        mutate_lane(input, output, ifft, axis)
    } else {
        mutate_lane(input, output, fft, axis)
    }
}

fn _fftnd<T: FFTnum+From<u32>, D: Dimension>(input: &mut ArrayViewMut<Complex<T>, D>, output: &mut ArrayViewMut<Complex<T>, D>, axes: &[usize], inverse: bool) {
    let len = axes.len();
    for (i, &axis) in axes.iter().enumerate() {
        _fftn(input, output, axis, inverse);
        if i < len - 1 {
            let mut outrows = output.genrows_mut().into_iter();
            for mut row in input.genrows_mut() {
                let mut outrow = outrows.next().unwrap();
                row.as_slice_mut().unwrap().copy_from_slice(outrow.as_slice_mut().unwrap());
            }
        }
    }
}


fn mutate_lane<T: Zero + Clone, D: Dimension>(input: &mut ArrayViewMut<T, D>, output: &mut ArrayViewMut<T, D>, f: fn(&mut [T], &mut [T]) -> (), axis: usize) {
    if axis > 0 {
        input.swap_axes(0, axis);
        output.swap_axes(0, axis);
        {
            let mut outrows = output.genrows_mut().into_iter();
            for row in input.genrows_mut() {
                let mut outrow = outrows.next().unwrap();
                let mut vec = row.to_vec();
                let mut out = vec![Zero::zero(); outrow.len()];
                f(&mut vec, &mut out);
                for i in 0..outrow.len() {
                    outrow[i] = out.remove(0);
                }
            }
        }
        input.swap_axes(0, axis);
        output.swap_axes(0, axis);
    } else {
        let mut outrows = output.genrows_mut().into_iter();
        for mut row in input.genrows_mut() {
            let mut outrow = outrows.next().unwrap();
            f(&mut row.as_slice_mut().unwrap(), &mut outrow.as_slice_mut().unwrap());
        }
    }
}

fn _fft<T: FFTnum>(input: &mut [Complex<T>], output: &mut [Complex<T>], inverse: bool) {
    let mut planner = FFTplanner::new(inverse);
    let len = input.len();
    let fft = planner.plan_fft(len);
    fft.process(input, output);
}

pub fn fft<T: FFTnum>(input: &mut [Complex<T>], output: &mut [Complex<T>]) {
    _fft(input, output, false);
}

pub fn ifft<T: FFTnum + From<u32>>(input: &mut [Complex<T>], output: &mut [Complex<T>]) {
    _fft(input, output, true);
    for v in output.iter_mut() {
        *v = v.unscale(T::from(input.len() as u32));
    }
}
