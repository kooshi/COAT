#include "include/fp8_adamw.h"
#include "include/fp8_adamw_expand.h"
#include "include/fp8_muon.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("fp8_adamw_step", &FP8_AdamW, "Update the quantized AdamW optimizer states");
    m.def("fp8_adamw_expand_step", &FP8_AdamW_expand, "Update the quantized AdamW optimizer states with polynomial range expansion");
    m.def("fp8_muon_step", &FP8_Muon, "Update FP8-quantized Muon momentum buffer (simple absmax scaling)");
    m.def("fp8_muon_expand_step", &FP8_Muon_expand, "Update FP8-quantized Muon momentum buffer with COAT polynomial range expansion");
}
