use crate::multimerge::{block_offsets, bottom_up_merge_kway, get_leaf_size};
use rayon::prelude::*;

pub const MIN_RUN: usize = 32;

pub const CHUNK: usize = 65_536;

#[inline]
fn insercao_binaria<T: Ord + Copy>(arr: &mut [T], ja_ordenado: usize) {
    for i in ja_ordenado.max(1)..arr.len() {
        let x = arr[i];

        let pos = arr[..i].partition_point(|y| *y <= x);
        if pos < i {
            arr[pos..=i].rotate_right(1);
        }
    }
}

pub fn build_min_runs<T: Ord + Copy + Send + Sync>(arr: &mut [T], min_run: usize) -> Vec<i64> {
    arr.par_chunks_mut(CHUNK)
        .map(|c| {
            let n = c.len();
            let mut meta: Vec<i64> = Vec::with_capacity(n / min_run + 2);
            let mut head = 0usize;
            while head < n {
                let mut tail = head + 1;
                if tail < n {
                    if c[head] <= c[tail] {
                        while tail < n && c[tail - 1] <= c[tail] {
                            tail += 1;
                        }
                    } else {
                        while tail < n && c[tail - 1] > c[tail] {
                            tail += 1;
                        }
                        c[head..tail].reverse();
                    }
                }
                let alvo = (head + min_run).min(n);
                if tail < alvo {
                    insercao_binaria(&mut c[head..alvo], tail - head);
                    tail = alvo;
                }
                meta.push((tail - head) as i64);
                head = tail;
            }
            meta
        })
        .reduce(Vec::new, |mut l, r| {
            l.extend_from_slice(&r);
            l
        })
}

pub fn sort_caotico<T: Ord + Copy + Send + Sync>(arr: &mut [T], min_run: usize) {
    let n = arr.len();
    let leaf_size = get_leaf_size::<T>();
    if n <= leaf_size {
        arr.sort();
        return;
    }
    let metadata = build_min_runs(arr, min_run);
    if metadata.len() == 1 {
        return;
    }
    let offsets = block_offsets(&metadata);
    let mut buffer: Vec<T> = vec![arr[0]; n];
    bottom_up_merge_kway(arr, &mut buffer, &metadata, &offsets, leaf_size, false);
}

pub fn build_blocos<T: Ord + Copy + Send + Sync>(arr: &mut [T], bloco: usize) -> Vec<i64> {
    let n = arr.len();
    arr.par_chunks_mut(bloco).for_each(|c| c.sort());
    let cheios = n / bloco;
    let resto = n % bloco;
    let mut meta: Vec<i64> = vec![bloco as i64; cheios];
    if resto > 0 { meta.push(resto as i64); }
    meta
}

pub fn sort_blocos<T: Ord + Copy + Send + Sync>(arr: &mut [T], bloco: usize) {
    let n = arr.len();
    let leaf_size = get_leaf_size::<T>();
    if n <= leaf_size { arr.sort(); return; }
    let metadata = build_blocos(arr, bloco);
    if metadata.len() == 1 { return; }
    let offsets = block_offsets(&metadata);
    let mut buffer: Vec<T> = vec![arr[0]; n];
    bottom_up_merge_kway(arr, &mut buffer, &metadata, &offsets, leaf_size, false);
}

pub fn cronometra<T: Ord + Copy + Send + Sync>(arr: &mut [T], bloco: usize) -> (f64, f64, usize) {
    use std::time::Instant;
    let n = arr.len();
    let leaf_size = get_leaf_size::<T>();
    let t = Instant::now();
    let metadata = build_blocos(arr, bloco);
    let t1 = t.elapsed().as_secs_f64() * 1e3;
    let offsets = block_offsets(&metadata);
    let mut buffer: Vec<T> = vec![arr[0]; n];
    let t = Instant::now();
    bottom_up_merge_kway(arr, &mut buffer, &metadata, &offsets, leaf_size, false);
    (t1, t.elapsed().as_secs_f64() * 1e3, metadata.len())
}

pub fn fase1_bench<T: Ord + Copy + Send + Sync>(arr: &mut [T], bloco: usize, estavel: bool) -> f64 {
    let t = std::time::Instant::now();
    if estavel {
        arr.par_chunks_mut(bloco).for_each(|c| c.sort());
    } else {
        arr.par_chunks_mut(bloco).for_each(|c| c.sort_unstable());
    }
    t.elapsed().as_secs_f64() * 1e3
}
