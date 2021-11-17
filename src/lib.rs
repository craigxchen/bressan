use std::collections::HashMap;
use std::cmp::min;

fn wedge_flow(z: f64) -> f64 {
    assert!((0.0 <= z) && (z <= 1.0));
    (1.0-4.0*(z-0.5).abs()).rem_euclid(1.0)
}

fn flow_map(x: f64, y: f64) -> (f64, f64) {
    let x_new: f64 = (x+wedge_flow(y)).rem_euclid(1.0);
    (x_new, (y+wedge_flow(x_new)).rem_euclid(1.0))
}
#[no_mangle]
pub extern "C" fn compute_period(x: f64, y: f64) -> i32 {
    let (mut x_new, mut y_new) = flow_map(x,y);
    let mut t = 1;
    while (x_new != x) || (y_new != y) {
        t += 1;
        let (_x_new, _y_new) = flow_map(x_new, y_new);
        x_new = _x_new;
        y_new = _y_new;
    }
    t
}

#[no_mangle]
pub extern "C" fn compute_min_period(k: i32) -> i32 {
    let mut res = 2_i32.pow(k as u32);
    let lim: f64 = 2.0_f64.powi(k);
    for i in (1..(lim as usize)).step_by(2) {
        for j in (1..(lim as usize)).step_by(2) {
            let (x, y) = ((i as f64) / lim, (j as f64) / lim);
            let mut t = 1;
            let (mut x_new, mut y_new) = flow_map(x,y);
            while (x_new != x) || (y_new != y) {
                t += 1;
                let (_x_new, _y_new) = flow_map(x_new, y_new);
                x_new = _x_new;
                y_new = _y_new;
                if t > res { break }
            }
            res = min(res, t);
        }
    }
    res
}

pub fn find_keys_for_value<'a, T, K: std::cmp::PartialEq>(
    map: &'a HashMap<T, K>,
    value: &K,
) -> Vec<&'a T> {
    map.iter()
        .filter_map(|(key, val)| if val == value { Some(key) } else { None })
        .collect()
}