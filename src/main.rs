use dyadic_periods::{
    find_keys_for_value,
    compute_period,
    compute_min_period,
};
use rayon::prelude::*;
use std::collections::HashMap;

fn main() {
    let k: i32 = 18;
    // let lim: f64 = 2.0_f64.powi(k);
    // let res: HashMap<_,i32> = (1..(lim as usize))
    //     .into_par_iter().step_by(2)
    //     .flat_map(|i| (1..(lim as usize)).into_par_iter().step_by(2).map(move |j| (i,j)))
    //     .map(|(i,j)| {
    //         ((i,j), compute_period( i as f64 / lim, j as f64 / lim))
    //     }).collect();
    // let min_period = res.values().min().unwrap();
    // let points = find_keys_for_value(&res, min_period);
    // println!("k={}: min_period={}", k, min_period);
    // // dbg!(points);
    let test = compute_min_period(k);
    println!("k={}, min_period={}",k,test);
}

