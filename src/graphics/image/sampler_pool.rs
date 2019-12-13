
use std::{
    mem::transmute,
    sync::Arc,
    collections::HashMap,
    hash::{Hash, Hasher},
};
use vulkano::{
    device::Device,
    sampler::{ SamplerAddressMode, MipmapMode, Filter, Sampler, SamplerCreationError },
};

fn hash_f32<H: Hasher>(f: f32, state: &mut H) {
    #![feature(float_to_from_bytes)]
    let bytes = f.to_be_bytes();
    bytes.hash(state);
}

#[derive(Clone)]
pub struct SamplerParams {
    mag_filter: Filter,
    min_filter: Filter,
    mipmap_mode: MipmapMode,
    u_addr: SamplerAddressMode,
    v_addr: SamplerAddressMode,
    w_addr: SamplerAddressMode,
    mip_lod_bias: f32,
    max_anisotropy: f32,
    min_lod: f32,
    max_lod: f32,
}
impl SamplerParams {
    pub fn simple_repeat() -> Self { Self {
        mag_filter: Filter::Linear,
        min_filter: Filter::Linear,
        mipmap_mode: MipmapMode::Linear,
        u_addr: SamplerAddressMode::Repeat,
        v_addr: SamplerAddressMode::Repeat,
        w_addr: SamplerAddressMode::Repeat,
        mip_lod_bias: 0.0,
        max_anisotropy: 1.0,
        min_lod: 0.0,
        max_lod: 1_000.0,
    } }

    fn generate_sampler(&self, device: Arc<Device>) -> Result<Arc<Sampler>, SamplerCreationError> {
        Sampler::new(device,
            self.mag_filter, self.min_filter, self.mipmap_mode, self.u_addr, self.v_addr,
            self.w_addr, self.mip_lod_bias, self.max_anisotropy, self.min_lod, self.max_lod
        )
    }
}
impl Hash for SamplerParams {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.mag_filter.hash(state);
        self.min_filter.hash(state);
        self.mipmap_mode.hash(state);
        self.u_addr.hash(state);
        self.v_addr.hash(state);
        self.w_addr.hash(state);
        hash_f32(self.mip_lod_bias, state);
        hash_f32(self.max_anisotropy, state);
        hash_f32(self.min_lod, state);
        hash_f32(self.max_lod, state);
    }
}
impl PartialEq for SamplerParams {
    fn eq(&self, o: &SamplerParams) -> bool {
        if self.mag_filter.ne(&o.mag_filter) { false }
        else if self.min_filter.ne(&o.min_filter) { false }
        else if self.mipmap_mode.ne(&o.mipmap_mode) { false }
        else if self.u_addr.ne(&o.u_addr) { false }
        else if self.v_addr.ne(&o.w_addr) { false }
        else if self.w_addr.ne(&o.v_addr) { false }
        else if self.mip_lod_bias.ne(&o.mip_lod_bias) { false }
        else if self.max_anisotropy.ne(&o.max_anisotropy) { false }
        else if self.min_lod.ne(&o.min_lod) { false }
        else if self.max_lod.ne(&o.max_lod) { false }
        else { true }
    }
}
impl Eq for SamplerParams {}

pub struct SamplerPool {
    device: Arc<Device>,
    samplers: HashMap<SamplerParams, Arc<Sampler>>,
}
impl SamplerPool {
    pub fn new(device: Arc<Device>) -> Self {
        Self {
            device,
            samplers: HashMap::new(),
        }
    }
    pub fn with_params(&mut self, params: SamplerParams) -> Arc<Sampler> {
        if let Some(sampler) = self.samplers.get(&params) {
            sampler.clone()
        } else {
            // TODO: Propagate error in a wrapper
            let sampler = params.generate_sampler(self.device.clone()).unwrap();
            self.samplers.insert(params, sampler.clone());
            sampler
        }
    }
}
