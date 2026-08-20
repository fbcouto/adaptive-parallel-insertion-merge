pub mod multimerge;
pub mod pim_kernel;
pub mod cut;
pub mod despacho;

pub mod amostra;

use rayon::prelude::*;
use std::sync::atomic::Ordering::Relaxed;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CustoComparacao {
    Embutida,
    Indireta,
}

#[derive(Clone, Copy, Debug)]
pub struct PimConfig {
    pub workers: usize,

    pub l3_bytes: usize,

    pub memory_limit_bytes: Option<usize>,
    pub custo_comparacao: CustoComparacao,

    pub min_gallop: usize,

    pub min_segmento_pway: usize,

    pub tarefas_por_worker: usize,

    pub bloco_override: Option<usize>,

    pub janelas_amostra: usize,

    pub tamanho_janela_amostra: usize,

    pub run_curto: usize,
    pub run_medio_minimo: usize,
    pub runs_via_kway: bool,
    pub folha_override: usize,
}

impl Default for PimConfig {
    fn default() -> Self {
        Self {
            workers: rayon::current_num_threads().max(1),
            l3_bytes: 48 << 20,
            memory_limit_bytes: None,
            custo_comparacao: CustoComparacao::Embutida,

            min_gallop: 24,
            min_segmento_pway: crate::cut::MIN_SEG,
            tarefas_por_worker: crate::cut::TAREFAS_POR_THREAD,
            bloco_override: None,
            janelas_amostra: 8,
            tamanho_janela_amostra: 96,
            run_curto: 128,
            run_medio_minimo: 16,
            runs_via_kway: true,
            folha_override: 0,
        }
    }
}

impl PimConfig {
    pub fn comparacao_indireta(mut self) -> Self {
        self.custo_comparacao = CustoComparacao::Indireta;
        self.min_gallop = 7;
        self.min_segmento_pway = 2_048;
        self
    }

    #[inline]
    fn workers_efetivos(&self) -> usize { self.workers.max(1) }
}

#[inline]
fn tamanho_folha_config<T>(config: PimConfig) -> usize {
    if config.folha_override > 0 {
        return config.folha_override;
    }
    let por_l1 = 32_768 / std::mem::size_of::<T>().max(1);
    let piso = match config.custo_comparacao {
        CustoComparacao::Embutida => 512,
        CustoComparacao::Indireta => 256,
    };
    por_l1.clamp(piso, 4_096)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PimError {
    MemoriaInsuficiente { necessario: usize, limite: usize },
    FalhaAoReservarMemoria { necessario: usize },
}

pub struct PimExecutor {
    pool: rayon::ThreadPool,
    workers: usize,
}

impl PimExecutor {
    pub fn new(workers: usize) -> Result<Self, rayon::ThreadPoolBuildError> {
        let workers = workers.max(1);
        let pool = rayon::ThreadPoolBuilder::new().num_threads(workers).build()?;
        Ok(Self { pool, workers })
    }

    pub fn try_sort<T: Ord + Copy + Send + Sync>(
        &self,
        arr: &mut [T],
        mut config: PimConfig,
    ) -> Result<(), PimError> {
        config.workers = config.workers_efetivos().min(self.workers);
        self.pool.install(|| try_pim_sort_with_config(arr, config))
    }

    pub fn sort<T: Ord + Copy + Send + Sync>(&self, arr: &mut [T], config: PimConfig) {
        if let Err(erro) = self.try_sort(arr, config) {
            panic!("PIM nao conseguiu reservar a memoria auxiliar: {erro:?}");
        }
    }

    #[inline]
    pub fn workers(&self) -> usize { self.workers }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PerfilEntrada {
    RunsLongos,
    RunsCurtos,
    Caotica,
}

fn classifica_entrada<T: Ord>(arr: &[T], config: PimConfig) -> PerfilEntrada {
    let comparacoes_disponiveis = arr.len().saturating_sub(1);
    if comparacoes_disponiveis < 32 {
        return PerfilEntrada::RunsLongos;
    }

    let janelas = config.janelas_amostra.clamp(3, 32);
    let largura = config
        .tamanho_janela_amostra
        .clamp(16, comparacoes_disponiveis);
    let mut mudancas = 0usize;
    let mut comparacoes = 0usize;
    let mut trechos = 0usize;

    for janela in 0..janelas {
        let inicio = if janelas == 1 {
            0
        } else {
            janela * (comparacoes_disponiveis - largura) / (janelas - 1)
        };
        let fim = inicio + largura;
        let mut ascendente = arr[inicio] <= arr[inicio + 1];
        let mut tamanho_run = 1usize;

        for i in (inicio + 1)..fim {
            let atual = arr[i] <= arr[i + 1];
            comparacoes += 1;
            if atual == ascendente {
                tamanho_run += 1;
            } else {
                mudancas += 1;
                trechos += 1;
                tamanho_run = 1;
                ascendente = atual;
            }
        }
        let _ = tamanho_run;
        trechos += 1;
    }

    if mudancas == 0 {
        PerfilEntrada::RunsLongos
    } else if mudancas.saturating_mul(100) >= comparacoes.saturating_mul(18) {
        PerfilEntrada::Caotica
    } else {
        let media_run = comparacoes / trechos.max(1);
        if media_run <= config.run_curto {
            PerfilEntrada::RunsCurtos
        } else {
            PerfilEntrada::RunsLongos
        }
    }
}

pub fn detect_global_trend<T: Ord + Sync>(arr: &[T]) -> Vec<i64> {
    let n = arr.len();
    if n <= 1 { return if n == 1 { vec![1] } else { Vec::new() }; }
    let macro_slice_len = 32_768;
    let macro_step = macro_slice_len - 1;
    if n <= macro_slice_len { return process_macro_block(arr); }

    let num_macro_blocks = (n + macro_step - 1) / macro_step;
    (0..num_macro_blocks).into_par_iter().map(|i| {
        let start = i * macro_step;
        let end = std::cmp::min(start + macro_slice_len, n);
        process_macro_block(&arr[start..end])
    }).reduce(|| Vec::new(), merge_metadata_pure)
}

fn process_macro_block<T: Ord + Sync>(arr: &[T]) -> Vec<i64> {
    let n = arr.len();
    let micro_slice_len = 512;
    let micro_step = micro_slice_len - 1;
    if n <= micro_slice_len { return generate_sequential_metadata(arr); }

    let num_micro_blocks = (n + micro_step - 1) / micro_step;
    (0..num_micro_blocks).into_par_iter().map(|i| {
        let start = i * micro_step;
        let end = std::cmp::min(start + micro_slice_len, n);
        generate_sequential_metadata(&arr[start..end])
    }).reduce(|| Vec::new(), merge_metadata_pure)
}

fn merge_metadata_pure(mut left: Vec<i64>, right: Vec<i64>) -> Vec<i64> {
    if left.is_empty() { return right; }
    if right.is_empty() { return left; }
    let last_left = left.pop().unwrap();
    let first_right = right[0];

    if last_left.unsigned_abs() == 1 { left.extend_from_slice(&right); return left; }
    if first_right.unsigned_abs() == 1 { left.push(last_left); left.extend_from_slice(&right[1..]); return left; }

    let left_asc = last_left > 0;
    let right_asc = first_right > 0;
    if left_asc == right_asc {
        let sign = if left_asc { 1 } else { -1 };
        left.push((last_left.unsigned_abs() + first_right.unsigned_abs() - 1) as i64 * sign);
        left.extend_from_slice(&right[1..]);
    } else {
        left.push(last_left);
        let sign = if right_asc { 1 } else { -1 };
        left.push((first_right.unsigned_abs() - 1) as i64 * sign);
        left.extend_from_slice(&right[1..]);
    }
    left
}

fn generate_sequential_metadata<T: Ord>(arr: &[T]) -> Vec<i64> {
    let n = arr.len();
    if n == 0 { return Vec::new(); }
    if n == 1 { return vec![1]; }
    let mut metadata = Vec::with_capacity(n / 64);
    let mut head = 0;
    while head < n - 1 {
        let mut tail = head + 1;
        if arr[head] <= arr[tail] {
            while tail < n && arr[tail - 1] <= arr[tail] { tail += 1; }
            metadata.push((tail - head) as i64);
        } else {
            while tail < n && arr[tail - 1] > arr[tail] { tail += 1; }
            metadata.push(-((tail - head) as i64));
        }
        head = tail;
    }
    if head == n - 1 { metadata.push(1); }
    metadata
}

#[inline]
fn merge_front<T: Ord + Copy>(a: &[T], b: &[T], dest: &mut [T]) {
    let (mut i, mut j) = (0, 0);
    for slot in dest.iter_mut() {
        if j >= b.len() || (i < a.len() && a[i] <= b[j]) {
            *slot = a[i]; i += 1;
        } else {
            *slot = b[j]; j += 1;
        }
    }
}

#[inline]
#[allow(dead_code)]
fn merge_back<T: Ord + Copy>(a: &[T], b: &[T], dest: &mut [T]) {
    let (mut i, mut j) = (a.len(), b.len());
    for slot in dest.iter_mut().rev() {
        if i == 0 || (j > 0 && b[j - 1] >= a[i - 1]) {
            j -= 1; *slot = b[j];
        } else {
            i -= 1; *slot = a[i];
        }
    }
}

pub static FOLHA_CHAMADAS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static FOLHA_MODO: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
pub static TRISECT_FRAC_MILI: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(400);
pub static TRISECT_MIN: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(512);

pub fn set_folha_modo(m: usize) {
    FOLHA_MODO.store(m, std::sync::atomic::Ordering::Relaxed);
}

pub fn set_trisect_frac(f: f64) {
    TRISECT_FRAC_MILI.store((f * 1000.0) as usize, std::sync::atomic::Ordering::Relaxed);
}
pub fn set_trisect_min(v: usize) {
    TRISECT_MIN.store(v, std::sync::atomic::Ordering::Relaxed);
}

pub fn set_folha_galopante(v: bool) {
    set_folha_modo(if v { 1 } else { 0 });
}

#[allow(dead_code)]
fn bidirectional_merge_ro<T: Ord + Copy + Send + Sync>(a: &[T], b: &[T], dest: &mut [T]) {
    FOLHA_CHAMADAS.fetch_add(1, Relaxed);
    let modo = FOLHA_MODO.load(Relaxed);

    if modo == 2 && dest.len() >= TRISECT_MIN.load(Relaxed) {
        let f = TRISECT_FRAC_MILI.load(Relaxed) as f64 / 1000.0;
        crate::cut::trisect_merge_gallop(a, b, dest, f);
        return;
    }

    let k = dest.len() / 2;
    let (df, db) = dest.split_at_mut(k);
    if modo >= 1 {
        rayon::join(
            || crate::pim_kernel::pim_front(a, b, df),
            || crate::pim_kernel::pim_back(a, b, db),
        );
    } else {
        rayon::join(|| merge_front(a, b, df), || merge_back(a, b, db));
    }
}

#[allow(dead_code)]
fn pim_parallel_merge<T: Ord + Copy + Send + Sync>(
    a: &[T], b: &[T], dest: &mut [T], is_a_left: bool, leaf_size: usize
) {
    let (n, m) = (a.len(), b.len());
    if n == 0 { dest.copy_from_slice(b); return; }
    if m == 0 { dest.copy_from_slice(a); return; }

    if n + m <= leaf_size || n < 3 {
        if is_a_left { bidirectional_merge_ro(a, b, dest); }
        else { bidirectional_merge_ro(b, a, dest); }
        return;
    }

    if n < m {
        pim_parallel_merge(b, a, dest, !is_a_left, leaf_size);
        return;
    }

    let mid_a = n / 2;
    let mid_b = if is_a_left { b.partition_point(|x| *x < a[mid_a]) } else { b.partition_point(|x| *x <= a[mid_a]) };
    let low_b = if is_a_left { b[..mid_b].partition_point(|x| *x < a[0]) } else { b[..mid_b].partition_point(|x| *x <= a[0]) };
    let high_b_rel = if is_a_left { b[mid_b..].partition_point(|x| *x < a[n - 1]) } else { b[mid_b..].partition_point(|x| *x <= a[n - 1]) };
    let high_b = mid_b + high_b_rel;

    let (dest_main, dest_top) = dest.split_at_mut(n + high_b - 1);
    dest_top[0] = a[n - 1];
    dest_top[1..].copy_from_slice(&b[high_b..]);

    let (dest_bottom, dest_right_rem) = dest_main.split_at_mut(mid_a + mid_b);
    dest_right_rem[0] = a[mid_a];
    let dest_right = &mut dest_right_rem[1..];

    let (dest_b_low, dest_left_rem) = dest_bottom.split_at_mut(low_b);
    dest_b_low.copy_from_slice(&b[..low_b]);
    dest_left_rem[0] = a[0];
    let dest_left = &mut dest_left_rem[1..];

    rayon::join(
        || pim_parallel_merge(&a[1..mid_a], &b[low_b..mid_b], dest_left, is_a_left, leaf_size),
        || pim_parallel_merge(&a[mid_a + 1..n - 1], &b[mid_b..high_b], dest_right, is_a_left, leaf_size)
    );
}

pub const KWAY_FANOUT: usize = 8;
#[allow(dead_code)]
const KWAY_MIN_BYTES: usize = 2 << 20;

#[inline]
#[allow(dead_code)]
fn tile_elems<T>() -> usize {
    const L1: usize = 32768;
    let e = std::mem::size_of::<T>().max(1);
    (L1 / (8 * e)).clamp(128, 2048)
}

#[allow(dead_code)]
fn multiseq_partition<T: Ord + Copy>(seq: &[&[T]], rank: usize, out: &mut [usize]) {
    let ns = seq.len();
    let mut lo = [0usize; KWAY_FANOUT];
    let mut hi = [0usize; KWAY_FANOUT];
    let mut plt = [0usize; KWAY_FANOUT];
    let mut ple = [0usize; KWAY_FANOUT];
    for i in 0..ns { hi[i] = seq[i].len(); }

    loop {
        let mut m = usize::MAX;
        let mut w = 0usize;
        for i in 0..ns {
            if hi[i] - lo[i] > w { w = hi[i] - lo[i]; m = i; }
        }
        if m == usize::MAX { out[..ns].copy_from_slice(&lo[..ns]); return; }

        let p = seq[m][lo[m] + (hi[m] - lo[m]) / 2];
        let (mut lt, mut le) = (0usize, 0usize);
        for i in 0..ns {
            plt[i] = seq[i].partition_point(|x| *x < p);
            ple[i] = seq[i].partition_point(|x| *x <= p);
            lt += plt[i]; le += ple[i];
        }

        if lt <= rank && rank <= le {
            let mut need = rank - lt;
            for i in 0..ns {
                let take = (ple[i] - plt[i]).min(need);
                out[i] = plt[i] + take; need -= take;
            }
            return;
        }
        if lt > rank {
            for i in 0..ns { if plt[i] < hi[i] { hi[i] = plt[i]; } }
        } else {
            for i in 0..ns { if ple[i] > lo[i] { lo[i] = ple[i]; } }
        }
    }
}

#[allow(dead_code)]
fn merge_level<T: Ord + Copy>(src: &[T], segs: &[(usize, usize)], dst: &mut [T]) -> ([(usize, usize); KWAY_FANOUT], usize) {
    let mut out = [(0usize, 0usize); KWAY_FANOUT];
    let mut cnt = 0usize;
    let mut off = 0usize;
    let mut i = 0usize;
    while i < segs.len() {
        if i + 1 == segs.len() {
            let (o, l) = segs[i];
            dst[off..off + l].copy_from_slice(&src[o..o + l]);
            out[cnt] = (off, l); off += l;
        } else {
            let (o1, l1) = segs[i];
            let (_, l2) = segs[i + 1];
            let l = l1 + l2;
            let (left, right) = src[o1..o1 + l].split_at(l1);
            merge_front(left, right, &mut dst[off..off + l]);
            out[cnt] = (off, l); off += l;
        }
        cnt += 1; i += 2;
    }
    (out, cnt)
}

#[allow(dead_code)]
fn kway_merge_tiled<T: Ord + Copy>(runs: &[&[T]], dest: &mut [T]) {
    let ns = runs.len();
    let total = dest.len();
    if ns == 1 { dest.copy_from_slice(&runs[0][..total]); return; }
    if ns == 2 { merge_front(runs[0], runs[1], dest); return; }

    let tile = tile_elems::<T>();
    let filler = runs.iter().find(|r| !r.is_empty()).map(|r| r[0]).unwrap();
    let mut scratch = vec![filler; 2 * tile];
    let (s1, s2) = scratch.split_at_mut(tile);

    let mut cur = [0usize; KWAY_FANOUT];
    let mut done = 0usize;

    while done < total {
        let want = tile.min(total - done);
        let mut win: [&[T]; KWAY_FANOUT] = [&[]; KWAY_FANOUT];
        for i in 0..ns {
            let rem = &runs[i][cur[i]..];
            win[i] = &rem[..rem.len().min(want)];
        }
        let mut cut = [0usize; KWAY_FANOUT];
        multiseq_partition(&win[..ns], want, &mut cut[..ns]);

        let mut segs = [(0usize, 0usize); KWAY_FANOUT];
        let mut cnt = 0usize;
        let mut off = 0usize;
        let mut i = 0usize;
        while i < ns {
            if i + 1 == ns {
                let l = cut[i];
                s1[off..off + l].copy_from_slice(&win[i][..l]);
                segs[cnt] = (off, l); off += l;
            } else {
                let l = cut[i] + cut[i + 1];
                merge_front(&win[i][..cut[i]], &win[i + 1][..cut[i + 1]], &mut s1[off..off + l]);
                segs[cnt] = (off, l); off += l;
            }
            cnt += 1; i += 2;
        }

        let out_slice = &mut dest[done..done + want];

        if cnt > 2 {
            let (segs2, cnt2) = merge_level(&s1[..want], &segs[..cnt], s2);
            if cnt2 == 2 {
                let (o1, l1) = segs2[0]; let (_, l2) = segs2[1];
                let (left, right) = s2[o1..o1 + l1 + l2].split_at(l1);
                merge_front(left, right, out_slice);
            } else {
                let (o, l) = segs2[0]; out_slice.copy_from_slice(&s2[o..o + l]);
            }
        } else if cnt == 2 {
            let (o1, l1) = segs[0]; let (_, l2) = segs[1];
            let (left, right) = s1[o1..o1 + l1 + l2].split_at(l1);
            merge_front(left, right, out_slice);
        } else {
            let (o, l) = segs[0]; out_slice.copy_from_slice(&s1[o..o + l]);
        }
        for i in 0..ns { cur[i] += cut[i]; }
        done += want;
    }
}

#[allow(dead_code)]
fn parallel_kway_pim_merge<T: Ord + Copy + Send + Sync>(runs: &[&[T]], dest: &mut [T], leaf_size: usize) {
    let ns = runs.len();
    let total = dest.len();

    if ns == 1 { dest.copy_from_slice(&runs[0][..total]); return; }

    if ns == 2 {
        pim_parallel_merge(runs[0], runs[1], dest, true, leaf_size);
        return;
    }

    if total <= leaf_size.saturating_mul(16) {
        kway_merge_tiled(runs, dest);
        return;
    }

    let mut cut = [0usize; KWAY_FANOUT];
    multiseq_partition(runs, total / 2, &mut cut[..ns]);

    let mut left: [&[T]; KWAY_FANOUT] = [&[]; KWAY_FANOUT];
    let mut right: [&[T]; KWAY_FANOUT] = [&[]; KWAY_FANOUT];
    let mut ltot = 0usize;
    for i in 0..ns {
        let (l, r) = runs[i].split_at(cut[i]);
        left[i] = l; right[i] = r; ltot += cut[i];
    }
    let (dl, dr) = dest.split_at_mut(ltot);
    rayon::join(
        || parallel_kway_pim_merge(&left[..ns], dl, leaf_size),
        || parallel_kway_pim_merge(&right[..ns], dr, leaf_size),
    );
}

fn block_offsets(metadata: &[i64]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(metadata.len() + 1);
    let mut off = 0usize; offsets.push(0);
    for &m in metadata { off += m.unsigned_abs() as usize; offsets.push(off); }
    offsets
}

fn parallel_reverse<T: Send + Sync>(arr: &mut [T]) {
    let n = arr.len();
    if n <= 100_000 { arr.reverse(); return; }
    let mid = n / 2;
    let (left, right) = arr.split_at_mut(mid);
    left.par_iter_mut().zip(right.par_iter_mut().rev()).for_each(|(a, b)| std::mem::swap(a, b));
}

#[allow(dead_code)]
fn bottom_up_merge_pim_kway<T: Ord + Copy + Send + Sync>(
    v: &mut [T], buf: &mut [T], metadata: &[i64], offsets: &[usize], leaf_size: usize, into_buf: bool,
) {
    let num_blocks = metadata.len();

    if num_blocks == 1 {
        let is_desc = metadata[0] < 0; let n = v.len();
        if into_buf {
            if is_desc { for i in 0..n { buf[i] = v[n - 1 - i]; } } else { buf.copy_from_slice(v); }
        } else if is_desc { parallel_reverse(v); }
        return;
    }

    let base = offsets[0];
    let total = offsets[num_blocks] - base;

    if total * std::mem::size_of::<T>() <= KWAY_MIN_BYTES {
        let target = base + total / 2;
        let split_idx = offsets.partition_point(|&o| o < target).clamp(1, num_blocks - 1);
        let mid = offsets[split_idx] - base;

        let (left_meta, right_meta) = metadata.split_at(split_idx);
        let left_offsets = &offsets[..=split_idx]; let right_offsets = &offsets[split_idx..];

        let (v_l, v_r) = v.split_at_mut(mid); let (buf_l, buf_r) = buf.split_at_mut(mid);
        rayon::join(
            || bottom_up_merge_pim_kway(v_l, buf_l, left_meta, left_offsets, leaf_size, !into_buf),
            || bottom_up_merge_pim_kway(v_r, buf_r, right_meta, right_offsets, leaf_size, !into_buf),
        );

        if into_buf {
            if v[mid - 1] <= v[mid] { buf.copy_from_slice(v); }
            else { pim_parallel_merge(&v[..mid], &v[mid..], buf, true, leaf_size); }
        } else {
            if buf[mid - 1] <= buf[mid] { v.copy_from_slice(buf); }
            else { pim_parallel_merge(&buf[..mid], &buf[mid..], v, true, leaf_size); }
        }
        return;
    }

    let g = KWAY_FANOUT.min(num_blocks);
    let mut cm = [0usize; KWAY_FANOUT + 1];
    cm[g] = num_blocks;
    for i in 1..g {
        let target = base + total * i / g;
        let idx = offsets.partition_point(|&o| o < target);
        cm[i] = idx.clamp(cm[i - 1] + 1, num_blocks - (g - i));
    }

    {
        let mut vv: &mut [T] = v; let mut bb: &mut [T] = buf;
        let mut prev = 0usize;
        let mut parts = Vec::with_capacity(g);
        for i in 0..g {
            let e = offsets[cm[i + 1]] - base; let len = e - prev;
            let (vl, vr) = vv.split_at_mut(len); let (bl, br) = bb.split_at_mut(len);
            parts.push((vl, bl, cm[i], cm[i + 1]));
            vv = vr; bb = br; prev = e;
        }
        parts.into_par_iter().for_each(|(sv, sb, lo, hi)| {
            bottom_up_merge_pim_kway(sv, sb, &metadata[lo..hi], &offsets[lo..=hi], leaf_size, !into_buf);
        });
    }

    let (src, dst): (&[T], &mut [T]) = if into_buf { (v, buf) } else { (buf, v) };
    let mut runs: [&[T]; KWAY_FANOUT] = [&[]; KWAY_FANOUT];
    let mut ns = 0usize; let mut gs = 0usize;
    for i in 0..g {
        let e = offsets[cm[i + 1]] - base;
        let join_next = i + 1 < g && src[e - 1] <= src[e];
        if !join_next {
            runs[ns] = &src[gs..e]; ns += 1; gs = e;
        }
    }

    if ns == 1 { dst.copy_from_slice(src); return; }
    parallel_kway_pim_merge(&runs[..ns], dst, leaf_size);
}

#[inline]
fn workers_por_merge(workers: usize, merges_simultaneos: usize) -> usize {
    let workers = workers.max(1);
    let merges = merges_simultaneos.max(1);
    workers.saturating_add(merges - 1) / merges
}

pub const PWAY_MIN_TOTAL: usize = 128;

pub const PWAY_BLOCK_SIZE: usize = 1_920;

pub static PWAY_BLOCK_OVERRIDE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

pub static L3_BYTES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(48 << 20);

pub fn set_pway_block(v: usize) {
    PWAY_BLOCK_OVERRIDE.store(v, std::sync::atomic::Ordering::Relaxed);
}
pub fn set_l3_bytes(v: usize) {
    L3_BYTES.store(v.max(1 << 18), std::sync::atomic::Ordering::Relaxed);
}

pub fn bloco_aleatorio<T>(n: usize, threads: usize) -> usize {
    let ov = PWAY_BLOCK_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
    if ov > 0 {
        return ov.max(2);
    }
    let t = threads.max(1);
    let orcamento_serial = L3_BYTES.load(std::sync::atomic::Ordering::Relaxed);

    let orcamento_efetivo = if t == 1 {
        orcamento_serial
    } else {
        orcamento_serial / 6
    };
    let por_thread = orcamento_efetivo / t;
    let elem = std::mem::size_of::<T>().max(1);
    let b_cache = por_thread / (3 * elem / 2).max(1);

    let b_balanco = if t > 1 { n / (4 * t) } else { usize::MAX };
    b_cache.min(b_balanco).clamp(4096, 8 << 20)
}

#[inline]
fn bloco_aleatorio_config<T>(n: usize, config: PimConfig) -> usize {
    if let Some(bloco) = config.bloco_override {
        return bloco.max(2);
    }
    let t = config.workers_efetivos();
    let orcamento_efetivo = if t == 1 {
        config.l3_bytes
    } else {
        config.l3_bytes / 6
    };
    let por_thread = orcamento_efetivo / t;
    let elem = std::mem::size_of::<T>().max(1);
    let b_cache = por_thread / (3 * elem / 2).max(1);
    let b_balanco = if t > 1 { n / (4 * t) } else { usize::MAX };
    b_cache.min(b_balanco).clamp(4096, 8 << 20)
}

#[inline]
fn deve_usar_pway(total: usize, workers: usize, config: PimConfig) -> bool {
    workers >= 2 && total / config.min_segmento_pway.max(1) >= 2
}

#[inline]
fn copia_faixas_disjuntas<T: Ord + Copy>(a: &[T], b: &[T], dest: &mut [T]) -> bool {
    if a.is_empty() || b.is_empty() {
        if a.is_empty() { dest.copy_from_slice(b); } else { dest.copy_from_slice(a); }
        return true;
    }
    if a[a.len() - 1] <= b[0] {
        let (da, db) = dest.split_at_mut(a.len());
        da.copy_from_slice(a);
        db.copy_from_slice(b);
        true
    } else if b[b.len() - 1] < a[0] {
        let (db, da) = dest.split_at_mut(b.len());
        db.copy_from_slice(b);
        da.copy_from_slice(a);
        true
    } else {
        false
    }
}

#[inline]
fn merge_binario_adaptativo<T: Ord + Copy + Send + Sync>(
    a: &[T],
    b: &[T],
    dest: &mut [T],
    workers: usize,
    config: PimConfig,
    atalhos_de_faixa: bool,
) {
    if atalhos_de_faixa && copia_faixas_disjuntas(a, b, dest) {
        return;
    }

    if workers >= 2 && dest.len() >= 512 {
        let meio = dest.len() / 2;
        let (frente, tras) = dest.split_at_mut(meio);
        rayon::join(
            || crate::pim_kernel::pim_front_adaptativo(a, b, frente, config.min_gallop),
            || crate::pim_kernel::pim_back_adaptativo(a, b, tras, config.min_gallop),
        );
    } else {
        crate::pim_kernel::pim_front_adaptativo(a, b, dest, config.min_gallop);
    }
}

fn merge_pim_pway_dinamico<T: Ord + Copy + Send + Sync>(
    a: &[T],
    b: &[T],
    dest: &mut [T],
    workers: usize,
    config: PimConfig,
    atalhos_de_faixa: bool,
) {
    let total = a.len() + b.len();
    if atalhos_de_faixa && copia_faixas_disjuntas(a, b, dest) {
        return;
    }
    if deve_usar_pway(total, workers, config) {
        crate::cut::pway_merge_frente_auto_adaptativo(
            a,
            b,
            dest,
            workers,
            config.min_segmento_pway,
            config.tarefas_por_worker,
            config.min_gallop,
        );
    } else {
        merge_binario_adaptativo(a, b, dest, workers, config, false);
    }
}

#[inline]
fn merge_pim_sequencial<T: Ord + Copy>(a: &[T], b: &[T], dest: &mut [T]) {
    if FOLHA_MODO.load(std::sync::atomic::Ordering::Relaxed) >= 1 {
        crate::pim_kernel::pim_front(a, b, dest);
    } else {
        merge_front(a, b, dest);
    }
}

fn bottom_up_merge_pim_sequencial<T: Ord + Copy>(
    v: &mut [T],
    buf: &mut [T],
    metadata: &[i64],
    offsets: &[usize],
    into_buf: bool,
    costura_bordas: bool,
) {
    let num_blocks = metadata.len();
    if num_blocks == 1 {
        let is_desc = metadata[0] < 0;
        let n = v.len();
        if into_buf {
            if is_desc {
                for i in 0..n {
                    buf[i] = v[n - 1 - i];
                }
            } else {
                buf.copy_from_slice(v);
            }
        } else if is_desc {
            v.reverse();
        }
        return;
    }

    let base = offsets[0];
    let total = offsets[num_blocks] - base;
    let target = base + total / 2;
    let split_idx = offsets
        .partition_point(|&o| o < target)
        .clamp(1, num_blocks - 1);
    let mid = offsets[split_idx] - base;

    let (left_meta, right_meta) = metadata.split_at(split_idx);
    let left_offsets = &offsets[..=split_idx];
    let right_offsets = &offsets[split_idx..];
    let (v_left, v_right) = v.split_at_mut(mid);
    let (buf_left, buf_right) = buf.split_at_mut(mid);
    bottom_up_merge_pim_sequencial(
        v_left,
        buf_left,
        left_meta,
        left_offsets,
        !into_buf,
        costura_bordas,
    );
    bottom_up_merge_pim_sequencial(
        v_right,
        buf_right,
        right_meta,
        right_offsets,
        !into_buf,
        costura_bordas,
    );

    if into_buf {
        if costura_bordas && v[mid - 1] <= v[mid] {
            buf.copy_from_slice(v);
        } else {
            merge_pim_sequencial(&v[..mid], &v[mid..], buf);
        }
    } else if costura_bordas && buf[mid - 1] <= buf[mid] {
        v.copy_from_slice(buf);
    } else {
        merge_pim_sequencial(&buf[..mid], &buf[mid..], v);
    }
}

fn bottom_up_merge_pim_pway<T: Ord + Copy + Send + Sync>(
    v: &mut [T],
    buf: &mut [T],
    metadata: &[i64],
    offsets: &[usize],
    into_buf: bool,
    workers_total: usize,
    merges_simultaneos: usize,
    config: PimConfig,
    atalhos_de_faixa: bool,
) {
    let num_blocks = metadata.len();

    if num_blocks == 1 {
        let is_desc = metadata[0] < 0;
        let n = v.len();
        if into_buf {
            if is_desc {
                for i in 0..n {
                    buf[i] = v[n - 1 - i];
                }
            } else {
                buf.copy_from_slice(v);
            }
        } else if is_desc {
            parallel_reverse(v);
        }
        return;
    }

    let base = offsets[0];
    let total = offsets[num_blocks] - base;
    let target = base + total / 2;
    let split_idx = offsets
        .partition_point(|&o| o < target)
        .clamp(1, num_blocks - 1);
    let mid = offsets[split_idx] - base;

    let (left_meta, right_meta) = metadata.split_at(split_idx);
    let left_offsets = &offsets[..=split_idx];
    let right_offsets = &offsets[split_idx..];
    let (v_left, v_right) = v.split_at_mut(mid);
    let (buf_left, buf_right) = buf.split_at_mut(mid);
    let filhos = merges_simultaneos.saturating_mul(2);
    rayon::join(
        || bottom_up_merge_pim_pway(
            v_left,
            buf_left,
            left_meta,
            left_offsets,
            !into_buf,
            workers_total,
            filhos,
            config,
            atalhos_de_faixa,
        ),
        || bottom_up_merge_pim_pway(
            v_right,
            buf_right,
            right_meta,
            right_offsets,
            !into_buf,
            workers_total,
            filhos,
            config,
            atalhos_de_faixa,
        ),
    );

    let workers = workers_por_merge(workers_total, merges_simultaneos);
    if into_buf {
        merge_pim_pway_dinamico(
            &v[..mid],
            &v[mid..],
            buf,
            workers,
            config,
            atalhos_de_faixa,
        );
    } else {
        merge_pim_pway_dinamico(
            &buf[..mid],
            &buf[mid..],
            v,
            workers,
            config,
            atalhos_de_faixa,
        );
    }
}

fn ordena_blocos<T: Ord + Copy + Send + Sync>(
    arr: &mut [T],
    block_size: usize,
    par_sort_local: bool,
) -> Vec<i64> {
    if par_sort_local {
        arr.par_chunks_mut(block_size)
            .for_each(|bloco| bloco.par_sort());
    } else {
        arr.par_chunks_mut(block_size)
            .for_each(|bloco| bloco.sort());
    }

    let cheios = arr.len() / block_size;
    let resto = arr.len() % block_size;
    let mut metadata = vec![block_size as i64; cheios];
    if resto > 0 {
        metadata.push(resto as i64);
    }
    metadata
}

#[inline]

fn numero_workers_pim() -> usize {
    std::env::var("PIM_NUM_THREADS")
        .ok()
        .and_then(|valor| valor.parse::<usize>().ok())
        .filter(|&workers| workers > 0)
        .unwrap_or_else(|| rayon::current_num_threads().max(1))
}

fn ordena_blocos_com_threads_pim<T: Ord + Send>(
    arr: &mut [T],
    block_size: usize,
    workers_concedidos: usize,
) -> Vec<i64> {
    let len = arr.len();
    let num_blocos = len.div_ceil(block_size);
    if num_blocos == 0 {
        return Vec::new();
    }

    let workers = workers_concedidos.max(1).min(num_blocos);

    std::thread::scope(|escopo| {
        let mut restante = arr;
        let mut blocos_restantes = num_blocos;
        let mut workers_restantes = workers;

        while workers_restantes > 0 {
            let blocos_deste_worker = blocos_restantes.div_ceil(workers_restantes);
            let elementos_deste_worker = blocos_deste_worker
                .saturating_mul(block_size)
                .min(restante.len());
            let (faixa, proxima_faixa) = restante.split_at_mut(elementos_deste_worker);
            restante = proxima_faixa;

            escopo.spawn(move || {
                for bloco in faixa.chunks_mut(block_size) {
                    bloco.sort();
                }
            });

            blocos_restantes -= blocos_deste_worker;
            workers_restantes -= 1;
        }
    });

    let cheios = len / block_size;
    let resto = len % block_size;
    let mut metadata = vec![block_size as i64; cheios];
    if resto > 0 {
        metadata.push(resto as i64);
    }
    metadata
}

fn ordena_bloco_por_subsorts<T: Ord + Copy + Send + Sync>(
    bloco: &mut [T],
    sub_sort_size: usize,
) {
    if bloco.len() <= sub_sort_size {
        bloco.sort();
        return;
    }

    let mut metadata = Vec::with_capacity(bloco.len().div_ceil(sub_sort_size));
    for pequeno in bloco.chunks_mut(sub_sort_size) {
        pequeno.sort();
        metadata.push(pequeno.len() as i64);
    }

    let offsets = block_offsets(&metadata);
    let mut buffer = vec![bloco[0]; bloco.len()];

    bottom_up_merge_pim_sequencial(bloco, &mut buffer, &metadata, &offsets, false, true);
}

fn ordena_blocos_por_subsorts<T: Ord + Copy + Send + Sync>(
    arr: &mut [T],
    block_size: usize,
    sub_sort_size: usize,
) -> Vec<i64> {
    arr.par_chunks_mut(block_size)
        .for_each(|bloco| ordena_bloco_por_subsorts(bloco, sub_sort_size));

    let cheios = arr.len() / block_size;
    let resto = arr.len() % block_size;
    let mut metadata = vec![block_size as i64; cheios];
    if resto > 0 {
        metadata.push(resto as i64);
    }
    metadata
}

fn conf_compatibilidade() -> PimConfig {
    let mut config = PimConfig::default();
    config.workers = numero_workers_pim();
    config.l3_bytes = L3_BYTES.load(Relaxed);
    config.bloco_override = match PWAY_BLOCK_OVERRIDE.load(Relaxed) {
        0 => None,
        bloco => Some(bloco),
    };
    if FOLHA_MODO.load(Relaxed) >= 1 {
        config = config.comparacao_indireta();
    }
    config
}

fn reserva_buffer<T: Copy>(
    arr: &[T],
    config: PimConfig,
    extras_bytes: usize,
) -> Result<Vec<T>, PimError> {
    let necessario = arr
        .len()
        .checked_mul(std::mem::size_of::<T>())
        .and_then(|bytes| bytes.checked_add(extras_bytes))
        .unwrap_or(usize::MAX);
    if let Some(limite) = config.memory_limit_bytes {
        if necessario > limite {
            return Err(PimError::MemoriaInsuficiente { necessario, limite });
        }
    }

    let mut sonda: Vec<T> = Vec::new();
    sonda
        .try_reserve_exact(arr.len())
        .map_err(|_| PimError::FalhaAoReservarMemoria { necessario })?;
    drop(sonda);

    Ok(vec![arr[0]; arr.len()])
}

fn try_pim_sort_blocos<T: Ord + Copy + Send + Sync>(
    arr: &mut [T],
    config: PimConfig,
) -> Result<(), PimError> {
    let bloco = bloco_aleatorio_config::<T>(arr.len(), config);
    if arr.len() <= bloco {
        arr.sort();
        return Ok(());
    }

    let runs = arr.len().div_ceil(bloco);
    let extras = runs.saturating_mul(std::mem::size_of::<i64>() + std::mem::size_of::<usize>());
    let mut buffer = reserva_buffer(arr, config, extras)?;
    let metadata = ordena_blocos_com_threads_pim(arr, bloco, config.workers_efetivos());
    let offsets = block_offsets(&metadata);
    bottom_up_merge_pim_pway(
        arr,
        &mut buffer,
        &metadata,
        &offsets,
        false,
        config.workers_efetivos(),
        1,
        config,
        false,
    );
    Ok(())
}

fn try_pim_sort_runs<T: Ord + Copy + Send + Sync>(
    arr: &mut [T],
    config: PimConfig,
) -> Result<(), PimError> {
    let metadata = detect_global_trend(arr);
    try_pim_sort_runs_com_metadata(arr, config, metadata)
}

fn try_pim_sort_runs_com_metadata<T: Ord + Copy + Send + Sync>(
    arr: &mut [T],
    config: PimConfig,
    metadata: Vec<i64>,
) -> Result<(), PimError> {
    if metadata.len() == 1 {
        if metadata[0] < 0 {
            parallel_reverse(arr);
        }
        return Ok(());
    }

    let extras = metadata
        .len()
        .saturating_mul(std::mem::size_of::<i64>() + std::mem::size_of::<usize>());
    let mut buffer = reserva_buffer(arr, config, extras)?;
    let offsets = block_offsets(&metadata);
    if config.runs_via_kway {
        crate::multimerge::bottom_up_merge_kway(
            arr,
            &mut buffer,
            &metadata,
            &offsets,
            tamanho_folha_config::<T>(config),
            false,
        );
    } else {
        bottom_up_merge_pim_pway(
            arr,
            &mut buffer,
            &metadata,
            &offsets,
            false,
            config.workers_efetivos(),
            1,
            config,
            true,
        );
    }
    Ok(())
}

pub fn try_pim_sort_with_config<T: Ord + Copy + Send + Sync>(
    arr: &mut [T],
    config: PimConfig,
) -> Result<(), PimError> {
    if arr.len() <= 1 {
        return Ok(());
    }
    if arr.len() <= tamanho_folha_config::<T>(config) {
        arr.sort();
        return Ok(());
    }

    let perfil = classifica_entrada(arr, config);

    if perfil == PerfilEntrada::RunsCurtos && config.run_medio_minimo > 0 {
        let metadata = detect_global_trend(arr);
        let runs = metadata.len().max(1);
        if arr.len() / runs >= config.run_medio_minimo {
            return try_pim_sort_runs_com_metadata(arr, config, metadata);
        }
        return try_pim_sort_blocos(arr, config);
    }

    match perfil {
        PerfilEntrada::RunsLongos => try_pim_sort_runs(arr, config),
        PerfilEntrada::RunsCurtos | PerfilEntrada::Caotica => try_pim_sort_blocos(arr, config),
    }
}

pub fn pim_sort_with_config<T: Ord + Copy + Send + Sync>(arr: &mut [T], config: PimConfig) {
    if let Err(erro) = try_pim_sort_with_config(arr, config) {
        panic!("PIM nao conseguiu reservar a memoria auxiliar: {erro:?}");
    }
}

pub fn pim_sort<T: Ord + Copy + Send + Sync>(arr: &mut [T]) {
    pim_sort_with_config(arr, conf_compatibilidade());
}

pub fn pim_sort_pway<T: Ord + Copy + Send + Sync>(arr: &mut [T]) {
    pim_sort(arr);
}

pub fn pim_sort_pway_sem_escudo<T: Ord + Copy + Send + Sync>(arr: &mut [T]) {
    if let Err(erro) = try_pim_sort_blocos(arr, conf_compatibilidade()) {
        panic!("PIM nao conseguiu reservar a memoria auxiliar: {erro:?}");
    }
}

pub fn pim_sort_pway_blocos<T: Ord + Copy + Send + Sync>(
    arr: &mut [T],
    block_size: usize,
    costura_bordas: bool,
) {
    pim_sort_pway_blocos_interno(arr, block_size, costura_bordas, true);
}

pub fn pim_sort_pway_blocos_sort_local<T: Ord + Copy + Send + Sync>(
    arr: &mut [T],
    block_size: usize,
    costura_bordas: bool,
) {
    pim_sort_pway_blocos_interno(arr, block_size, costura_bordas, false);
}

pub fn pim_sort_pway_blocos_subsorts<T: Ord + Copy + Send + Sync>(
    arr: &mut [T],
    block_size: usize,
    sub_sort_size: usize,
    costura_bordas: bool,
) {
    let block_size = block_size.max(2);
    let sub_sort_size = sub_sort_size.clamp(2, block_size);
    if arr.len() <= block_size {
        ordena_bloco_por_subsorts(arr, sub_sort_size);
        return;
    }

    let metadata = ordena_blocos_por_subsorts(arr, block_size, sub_sort_size);
    let offsets = block_offsets(&metadata);
    let mut buffer = vec![arr[0]; arr.len()];
    bottom_up_merge_pim_pway(
        arr,
        &mut buffer,
        &metadata,
        &offsets,
        false,
        rayon::current_num_threads(),
        1,
        conf_compatibilidade(),
        costura_bordas,
    );
}

fn pim_sort_pway_blocos_interno<T: Ord + Copy + Send + Sync>(
    arr: &mut [T],
    block_size: usize,
    costura_bordas: bool,
    par_sort_local: bool,
) {
    let block_size = block_size.max(2);
    if arr.len() <= block_size {
        if par_sort_local {
            arr.par_sort();
        } else {
            arr.sort();
        }
        return;
    }

    let metadata = ordena_blocos(arr, block_size, par_sort_local);
    let offsets = block_offsets(&metadata);
    let mut buffer = vec![arr[0]; arr.len()];
    bottom_up_merge_pim_pway(
        arr,
        &mut buffer,
        &metadata,
        &offsets,
        false,
        rayon::current_num_threads(),
        1,
        conf_compatibilidade(),
        costura_bordas,
    );
}

pub fn pim_sort_pway_blocos_sem_costura<T: Ord + Copy + Send + Sync>(
    arr: &mut [T],
    block_size: usize,
) {
    pim_sort_pway_blocos(arr, block_size, false);
}

#[cfg(test)]
mod testes_pway_dinamico {
    use super::*;

    #[derive(Clone, Copy, Debug)]
    struct K {
        chave: u32,
        indice: u32,
    }

    impl PartialEq for K {
        fn eq(&self, other: &Self) -> bool { self.chave == other.chave }
    }
    impl Eq for K {}
    impl PartialOrd for K {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for K {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.chave.cmp(&other.chave)
        }
    }

    fn proximo(estado: &mut u64) -> u64 {
        *estado ^= *estado << 13;
        *estado ^= *estado >> 7;
        *estado ^= *estado << 17;
        *estado
    }

    #[test]
    fn pway_dinamico_sem_escudo_bate_sort_estavel_em_entrada_caotica() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        pool.install(|| {
            let mut estado = 0xD1B54A32D192ED03u64;
            let base: Vec<K> = (0..50_000u32)
                .map(|indice| K {
                    chave: (proximo(&mut estado) % 97) as u32,
                    indice,
                })
                .collect();
            let mut esperado = base.clone();
            esperado.sort();

            for (bloco, costura) in [(512, true), (512, false), (2_048, true)] {
                let mut entrada = base.clone();
                pim_sort_pway_blocos(&mut entrada, bloco, costura);
                assert_eq!(
                    entrada.iter().map(|x| (x.chave, x.indice)).collect::<Vec<_>>(),
                    esperado.iter().map(|x| (x.chave, x.indice)).collect::<Vec<_>>(),
                    "bloco={bloco}, costura={costura}",
                );
            }

            let mut entrada = base.clone();
            pim_sort_pway_blocos_sort_local(&mut entrada, 2_048, true);
            assert_eq!(
                entrada.iter().map(|x| (x.chave, x.indice)).collect::<Vec<_>>(),
                esperado.iter().map(|x| (x.chave, x.indice)).collect::<Vec<_>>(),
                "sort local",
            );

            let mut entrada = base.clone();
            pim_sort(&mut entrada);
            assert_eq!(
                entrada.iter().map(|x| (x.chave, x.indice)).collect::<Vec<_>>(),
                esperado.iter().map(|x| (x.chave, x.indice)).collect::<Vec<_>>(),
                "fallback aleatorio de pim_sort",
            );

            for sub_bloco in [32, 64, 128, 240, 480, 960] {
                let mut entrada = base.clone();
                pim_sort_pway_blocos_subsorts(&mut entrada, 1_920, sub_bloco, true);
                assert_eq!(
                    entrada.iter().map(|x| (x.chave, x.indice)).collect::<Vec<_>>(),
                    esperado.iter().map(|x| (x.chave, x.indice)).collect::<Vec<_>>(),
                    "sub-sort={sub_bloco}",
                );
            }
        });
    }

    #[test]
    fn pway_exige_workers_e_segmentos_suficientes() {
        let config = PimConfig::default();
        let minimo = config.min_segmento_pway * 2;
        assert!(!deve_usar_pway(minimo, 1, config));
        assert!(!deve_usar_pway(minimo - 1, 8, config));
        assert!(deve_usar_pway(minimo, 2, config));
    }

    #[test]
    fn classificador_separa_runs_longos_curtos_e_caos() {
        let config = PimConfig::default();
        let ordenado: Vec<u32> = (0..4_096).collect();
        assert_eq!(classifica_entrada(&ordenado, config), PerfilEntrada::RunsLongos);

        let serra: Vec<u32> = (0..4_096).map(|i| i % 50).collect();
        assert_eq!(classifica_entrada(&serra, config), PerfilEntrada::RunsCurtos);

        let mut estado = 0xD1B5_4A32_9C7E_1023u64;
        let aleatorio: Vec<u64> = (0..4_096).map(|_| proximo(&mut estado)).collect();
        assert_eq!(classifica_entrada(&aleatorio, config), PerfilEntrada::Caotica);
    }

    #[test]
    fn configuracao_preserva_estabilidade_em_runs_e_caos() {
        let mut estado = 0xA24B_1CD3_958E_F760u64;
        let mut config = PimConfig::default();
        config.workers = 2;
        config.bloco_override = Some(256);
        config.min_segmento_pway = 128;
        let executor = PimExecutor::new(2).unwrap();

        for forma in 0..3 {
            let mut entrada: Vec<K> = (0..20_000u32)
                .map(|indice| {
                    let chave = match forma {
                        0 => indice % 97,
                        1 => indice % 50,
                        _ => proximo(&mut estado) as u32 % 5_000,
                    };
                    K { chave, indice }
                })
                .collect();
            if forma == 0 {
                for i in (1..entrada.len()).step_by(2) {
                    entrada.swap(i - 1, i);
                }
            }

            let mut esperado = entrada.clone();
            esperado.sort();
            executor.try_sort(&mut entrada, config).unwrap();
            assert_eq!(
                entrada.iter().map(|x| (x.chave, x.indice)).collect::<Vec<_>>(),
                esperado.iter().map(|x| (x.chave, x.indice)).collect::<Vec<_>>(),
                "forma={forma}",
            );
        }
    }

    #[test]
    fn limite_de_memoria_e_recuperavel_antes_da_fase_de_blocos() {
        let mut estado = 0x45F8_71A9_BCE0_1234u64;
        let mut entrada: Vec<u64> = (0..8_192).map(|_| proximo(&mut estado)).collect();
        let original = entrada.clone();
        let mut config = PimConfig::default();
        config.bloco_override = Some(256);
        config.memory_limit_bytes = Some(64);

        assert!(matches!(
            try_pim_sort_with_config(&mut entrada, config),
            Err(PimError::MemoriaInsuficiente { .. })
        ));
        assert_eq!(entrada, original, "o erro nao pode deixar blocos parcialmente ordenados");
    }

    #[test]
    fn atalho_de_faixa_invertida_mantem_a_ordem_estavel() {
        let a = [K { chave: 10, indice: 0 }, K { chave: 11, indice: 1 }];
        let b = [K { chave: 1, indice: 2 }, K { chave: 2, indice: 3 }];
        let mut destino = [K { chave: 0, indice: 0 }; 4];
        assert!(copia_faixas_disjuntas(&a, &b, &mut destino));
        assert_eq!(
            destino.iter().map(|x| (x.chave, x.indice)).collect::<Vec<_>>(),
            vec![(1, 2), (2, 3), (10, 0), (11, 1)],
        );
    }
}
