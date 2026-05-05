// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// PPISP (Physically Plausible Image Signal Processing) SPG Shader
//
// Implements the ISP pipeline for USD RenderProducts:
// 1. Exposure compensation
// 2. Vignetting correction (per-channel)
// 3. Color correction via ZCA-based homography
// 4. Camera Response Function (per-channel, 4-param toe-shoulder curve)
//
// NOTE: All parameters use flat naming to match USD inputs: attributes (UsdShade-compatible).
// SPG requires the Slang struct field names to match USD input names.

struct PPISPParams
{
    // Exposure
    float exposureOffset;

    // Vignetting R channel
    float2 vignettingCenterR;
    float vignettingAlpha1R;
    float vignettingAlpha2R;
    float vignettingAlpha3R;

    // Vignetting G channel
    float2 vignettingCenterG;
    float vignettingAlpha1G;
    float vignettingAlpha2G;
    float vignettingAlpha3G;

    // Vignetting B channel
    float2 vignettingCenterB;
    float vignettingAlpha1B;
    float vignettingAlpha2B;
    float vignettingAlpha3B;

    // Color correction: 4 control-point latent offsets (Blue, Red, Green, Neutral)
    float2 colorLatentBlue;
    float2 colorLatentRed;
    float2 colorLatentGreen;
    float2 colorLatentNeutral;

    // CRF R channel (raw params: activations applied at runtime)
    float crfToeR;
    float crfShoulderR;
    float crfGammaR;
    float crfCenterR;

    // CRF G channel
    float crfToeG;
    float crfShoulderG;
    float crfGammaG;
    float crfCenterG;

    // CRF B channel
    float crfToeB;
    float crfShoulderB;
    float crfGammaB;
    float crfCenterB;
};

[[vk::binding(0, 1)]] ParameterBlock<PPISPParams> g_Params;
[[vk::binding(1, 1)]] Texture2D<float4> g_InTex;
[[vk::binding(2, 1)]] RWTexture2D<float4> g_OutTex;

// ZCA pinv 2x2 blocks (constant, matching ppisp_math.cuh COLOR_PINV_BLOCKS)
static const float2x2 ZCA_BLUE    = float2x2( 0.0480542, -0.0043631, -0.0043631,  0.0481283);
static const float2x2 ZCA_RED     = float2x2( 0.0580570, -0.0179872, -0.0179872,  0.0431061);
static const float2x2 ZCA_GREEN   = float2x2( 0.0433336, -0.0180537, -0.0180537,  0.0580500);
static const float2x2 ZCA_NEUTRAL = float2x2( 0.0128369, -0.0034654, -0.0034654,  0.0128158);

// Compute 3x3 homography from ZCA latent offsets (port of compute_homography from ppisp_math.cuh)
float3x3 computeHomography(float2 bLat, float2 rLat, float2 gLat, float2 nLat)
{
    float2 bd = mul(ZCA_BLUE,    bLat);
    float2 rd = mul(ZCA_RED,     rLat);
    float2 gd = mul(ZCA_GREEN,   gLat);
    float2 neutralDelta = mul(ZCA_NEUTRAL, nLat);

    // Target chromaticities: source + offset. Source = (r,g,I) for pure B,R,G,gray
    float3 tB    = float3(0.0 + bd.x, 0.0 + bd.y, 1.0);
    float3 tR    = float3(1.0 + rd.x, 0.0 + rd.y, 1.0);
    float3 tG    = float3(0.0 + gd.x, 1.0 + gd.y, 1.0);
    float3 tGray = float3(1.0 / 3.0 + neutralDelta.x, 1.0 / 3.0 + neutralDelta.y, 1.0);

    // T = [tB | tR | tG] as columns (row-major: row i = [tB[i], tR[i], tG[i]])
    float3x3 T = float3x3(tB.x, tR.x, tG.x,
                           tB.y, tR.y, tG.y,
                           tB.z, tR.z, tG.z);

    // Skew-symmetric matrix [tGray]_x
    float3x3 skew = float3x3(0.0,     -tGray.z,  tGray.y,
                              tGray.z,  0.0,      -tGray.x,
                             -tGray.y,  tGray.x,   0.0);

    float3x3 M = mul(skew, T);

    // Null-space vector via cross product of first two rows
    float3 r0 = M[0];
    float3 r1 = M[1];
    float3 r2 = M[2];

    float3 lam = cross(r0, r1);
    if (dot(lam, lam) < 1.0e-20)
    {
        lam = cross(r0, r2);
        if (dot(lam, lam) < 1.0e-20)
            lam = cross(r1, r2);
    }

    // S_inv = [[-1,-1,1],[1,0,0],[0,1,0]]
    float3x3 Sinv = float3x3(-1.0, -1.0, 1.0,
                               1.0,  0.0, 0.0,
                               0.0,  1.0, 0.0);

    // D = diag(lam)
    float3x3 D = float3x3(lam.x, 0.0,   0.0,
                           0.0,   lam.y, 0.0,
                           0.0,   0.0,   lam.z);

    // H = T * D * S_inv
    float3x3 H = mul(mul(T, D), Sinv);

    // Normalize so H[2][2] = 1
    float s = H[2][2];
    if (abs(s) > 1.0e-20)
        H = H * (1.0 / s);

    return H;
}

float applyVignetting(float value, float2 uv, float2 opticalCenter, float alpha1, float alpha2, float alpha3)
{
    float2 delta = uv - opticalCenter;
    float r2 = dot(delta, delta);

    float falloff = 1.0;
    float r2Pow = r2;
    falloff += alpha1 * r2Pow;
    r2Pow *= r2;
    falloff += alpha2 * r2Pow;
    r2Pow *= r2;
    falloff += alpha3 * r2Pow;

    falloff = clamp(falloff, 0.0, 1.0);
    return value * falloff;
}

float boundedSoftplus(float raw, float minValue)
{
    return minValue + log(1.0 + exp(raw));
}

float sigmoid(float raw)
{
    return 1.0 / (1.0 + exp(-raw));
}

// 4-param toe-shoulder CRF (port of apply_crf_ppisp from ppisp_math.cuh)
float applyCRF(float x, float toeRaw, float shoulderRaw, float gammaRaw, float centerRaw)
{
    x = clamp(x, 0.0, 1.0);

    float toe      = boundedSoftplus(toeRaw, 0.3);
    float shoulder = boundedSoftplus(shoulderRaw, 0.3);
    float gamma    = boundedSoftplus(gammaRaw, 0.1);
    float center   = sigmoid(centerRaw);

    // toe >= 0.3, shoulder >= 0.3, center in (0,1) — divisions are safe
    float lerpVal = (shoulder - toe) * center + toe;
    float a = (shoulder * center) / lerpVal;
    float b = 1.0 - a;

    float y;
    if (x <= center)
        y = a * pow(x / center, toe);
    else
        y = 1.0 - b * pow((1.0 - x) / (1.0 - center), shoulder);

    return pow(max(0.0, y), gamma);
}

float3 applyColorCorrection(float3 rgb, float3x3 H)
{
    float intensity = rgb.x + rgb.y + rgb.z;
    float3 rgi = float3(rgb.x, rgb.y, intensity);

    rgi = mul(H, rgi);

    rgi = rgi * (intensity / (rgi.z + 1.0e-5));
    return float3(rgi.x, rgi.y, rgi.z - rgi.x - rgi.y);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void ppispProcess(uint3 tid : SV_DispatchThreadID)
{
    uint w = 0, h = 0;
    g_InTex.GetDimensions(w, h);
    if (tid.x >= w || tid.y >= h)
        return;

    float4 pixel = g_InTex.Load(int3(tid.xy, 0));
    float3 rgb = pixel.rgb;

    // Normalize to [-0.5, 0.5] range based on max dimension (matching CUDA kernel)
    float maxRes = max(float(w), float(h));
    float2 uv = float2(tid.x + 0.5 - float(w) * 0.5, tid.y + 0.5 - float(h) * 0.5) / maxRes;

    // 1. Exposure
    rgb = rgb * exp2(g_Params.exposureOffset);

    // 2. Vignetting (per-channel)
    rgb.r = applyVignetting(rgb.r, uv, g_Params.vignettingCenterR,
                            g_Params.vignettingAlpha1R, g_Params.vignettingAlpha2R, g_Params.vignettingAlpha3R);
    rgb.g = applyVignetting(rgb.g, uv, g_Params.vignettingCenterG,
                            g_Params.vignettingAlpha1G, g_Params.vignettingAlpha2G, g_Params.vignettingAlpha3G);
    rgb.b = applyVignetting(rgb.b, uv, g_Params.vignettingCenterB,
                            g_Params.vignettingAlpha1B, g_Params.vignettingAlpha2B, g_Params.vignettingAlpha3B);

    // 3. Color correction (ZCA-based homography)
    float3x3 H = computeHomography(g_Params.colorLatentBlue, g_Params.colorLatentRed,
                                   g_Params.colorLatentGreen, g_Params.colorLatentNeutral);
    rgb = applyColorCorrection(rgb, H);

    // 4. CRF (per-channel, 4-param toe-shoulder)
    rgb.r = applyCRF(rgb.r, g_Params.crfToeR, g_Params.crfShoulderR, g_Params.crfGammaR, g_Params.crfCenterR);
    rgb.g = applyCRF(rgb.g, g_Params.crfToeG, g_Params.crfShoulderG, g_Params.crfGammaG, g_Params.crfCenterG);
    rgb.b = applyCRF(rgb.b, g_Params.crfToeB, g_Params.crfShoulderB, g_Params.crfGammaB, g_Params.crfCenterB);

    g_OutTex[tid.xy] = float4(saturate(rgb), 1.0);
}
