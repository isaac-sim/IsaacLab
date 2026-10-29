# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .. import mdp
from ..heterogeneous_tasks_env_cfg import VLABenchmarkEnvCfg
from .pick_place_000_env_cfg import PickPlace000EnvCfg
from .pick_place_001_env_cfg import PickPlace001EnvCfg
from .pick_place_002_env_cfg import PickPlace002EnvCfg
from .pick_place_003_env_cfg import PickPlace003EnvCfg
from .pick_place_004_env_cfg import PickPlace004EnvCfg
from .pick_place_005_env_cfg import PickPlace005EnvCfg
from .pick_place_006_env_cfg import PickPlace006EnvCfg
from .pick_place_007_env_cfg import PickPlace007EnvCfg
from .pick_place_008_env_cfg import PickPlace008EnvCfg
from .pick_place_009_env_cfg import PickPlace009EnvCfg
from .pick_place_010_env_cfg import PickPlace010EnvCfg
from .pick_place_011_env_cfg import PickPlace011EnvCfg
from .pick_place_012_env_cfg import PickPlace012EnvCfg
from .pick_place_013_env_cfg import PickPlace013EnvCfg
from .pick_place_014_env_cfg import PickPlace014EnvCfg
from .pick_place_015_env_cfg import PickPlace015EnvCfg
from .pick_place_016_env_cfg import PickPlace016EnvCfg
from .pick_place_017_env_cfg import PickPlace017EnvCfg
from .pick_place_018_env_cfg import PickPlace018EnvCfg
from .pick_place_019_env_cfg import PickPlace019EnvCfg
from .pick_place_020_env_cfg import PickPlace020EnvCfg
from .pick_place_021_env_cfg import PickPlace021EnvCfg
from .pick_place_022_env_cfg import PickPlace022EnvCfg
from .pick_place_023_env_cfg import PickPlace023EnvCfg
from .pick_place_024_env_cfg import PickPlace024EnvCfg
from .pick_place_025_env_cfg import PickPlace025EnvCfg
from .pick_place_026_env_cfg import PickPlace026EnvCfg
from .pick_place_027_env_cfg import PickPlace027EnvCfg
from .pick_place_028_env_cfg import PickPlace028EnvCfg
from .pick_place_029_env_cfg import PickPlace029EnvCfg
from .pick_place_030_env_cfg import PickPlace030EnvCfg
from .pick_place_031_env_cfg import PickPlace031EnvCfg
from .pick_place_032_env_cfg import PickPlace032EnvCfg
from .pick_place_033_env_cfg import PickPlace033EnvCfg
from .pick_place_034_env_cfg import PickPlace034EnvCfg
from .pick_place_035_env_cfg import PickPlace035EnvCfg
from .pick_place_036_env_cfg import PickPlace036EnvCfg
from .pick_place_037_env_cfg import PickPlace037EnvCfg
from .pick_place_038_env_cfg import PickPlace038EnvCfg
from .pick_place_039_env_cfg import PickPlace039EnvCfg
from .pick_place_040_env_cfg import PickPlace040EnvCfg
from .pick_place_041_env_cfg import PickPlace041EnvCfg
from .pick_place_042_env_cfg import PickPlace042EnvCfg
from .pick_place_043_env_cfg import PickPlace043EnvCfg
from .pick_place_044_env_cfg import PickPlace044EnvCfg
from .pick_place_045_env_cfg import PickPlace045EnvCfg
from .pick_place_046_env_cfg import PickPlace046EnvCfg
from .pick_place_047_env_cfg import PickPlace047EnvCfg
from .pick_place_048_env_cfg import PickPlace048EnvCfg
from .pick_place_049_env_cfg import PickPlace049EnvCfg
from .pick_place_050_env_cfg import PickPlace050EnvCfg
from .pick_place_051_env_cfg import PickPlace051EnvCfg
from .pick_place_052_env_cfg import PickPlace052EnvCfg
from .pick_place_053_env_cfg import PickPlace053EnvCfg
from .pick_place_054_env_cfg import PickPlace054EnvCfg
from .pick_place_055_env_cfg import PickPlace055EnvCfg
from .pick_place_056_env_cfg import PickPlace056EnvCfg
from .pick_place_057_env_cfg import PickPlace057EnvCfg
from .pick_place_058_env_cfg import PickPlace058EnvCfg
from .pick_place_059_env_cfg import PickPlace059EnvCfg
from .pick_place_060_env_cfg import PickPlace060EnvCfg
from .pick_place_061_env_cfg import PickPlace061EnvCfg
from .pick_place_062_env_cfg import PickPlace062EnvCfg
from .pick_place_063_env_cfg import PickPlace063EnvCfg
from .pick_place_064_env_cfg import PickPlace064EnvCfg
from .pick_place_065_env_cfg import PickPlace065EnvCfg
from .pick_place_066_env_cfg import PickPlace066EnvCfg
from .pick_place_067_env_cfg import PickPlace067EnvCfg
from .pick_place_068_env_cfg import PickPlace068EnvCfg
from .pick_place_069_env_cfg import PickPlace069EnvCfg
from .pick_place_070_env_cfg import PickPlace070EnvCfg
from .pick_place_071_env_cfg import PickPlace071EnvCfg
from .pick_place_072_env_cfg import PickPlace072EnvCfg
from .pick_place_073_env_cfg import PickPlace073EnvCfg
from .pick_place_074_env_cfg import PickPlace074EnvCfg
from .pick_place_075_env_cfg import PickPlace075EnvCfg
from .pick_place_076_env_cfg import PickPlace076EnvCfg
from .pick_place_077_env_cfg import PickPlace077EnvCfg
from .pick_place_078_env_cfg import PickPlace078EnvCfg
from .pick_place_079_env_cfg import PickPlace079EnvCfg
from .pick_place_080_env_cfg import PickPlace080EnvCfg
from .pick_place_081_env_cfg import PickPlace081EnvCfg
from .pick_place_082_env_cfg import PickPlace082EnvCfg
from .pick_place_083_env_cfg import PickPlace083EnvCfg
from .pick_place_084_env_cfg import PickPlace084EnvCfg
from .pick_place_085_env_cfg import PickPlace085EnvCfg
from .pick_place_086_env_cfg import PickPlace086EnvCfg
from .pick_place_087_env_cfg import PickPlace087EnvCfg
from .pick_place_088_env_cfg import PickPlace088EnvCfg
from .pick_place_089_env_cfg import PickPlace089EnvCfg
from .pick_place_090_env_cfg import PickPlace090EnvCfg
from .pick_place_091_env_cfg import PickPlace091EnvCfg
from .pick_place_092_env_cfg import PickPlace092EnvCfg
from .pick_place_093_env_cfg import PickPlace093EnvCfg
from .pick_place_094_env_cfg import PickPlace094EnvCfg
from .pick_place_095_env_cfg import PickPlace095EnvCfg
from .pick_place_096_env_cfg import PickPlace096EnvCfg
from .pick_place_097_env_cfg import PickPlace097EnvCfg
from .pick_place_098_env_cfg import PickPlace098EnvCfg
from .pick_place_099_env_cfg import PickPlace099EnvCfg
from .pick_place_100_env_cfg import PickPlace100EnvCfg
from .pick_place_101_env_cfg import PickPlace101EnvCfg
from .pick_place_102_env_cfg import PickPlace102EnvCfg
from .pick_place_103_env_cfg import PickPlace103EnvCfg
from .pick_place_104_env_cfg import PickPlace104EnvCfg
from .pick_place_105_env_cfg import PickPlace105EnvCfg
from .pick_place_106_env_cfg import PickPlace106EnvCfg
from .pick_place_107_env_cfg import PickPlace107EnvCfg
from .pick_place_108_env_cfg import PickPlace108EnvCfg
from .pick_place_109_env_cfg import PickPlace109EnvCfg
from .pick_place_110_env_cfg import PickPlace110EnvCfg
from .pick_place_111_env_cfg import PickPlace111EnvCfg
from .pick_place_112_env_cfg import PickPlace112EnvCfg
from .pick_place_113_env_cfg import PickPlace113EnvCfg
from .pick_place_114_env_cfg import PickPlace114EnvCfg
from .pick_place_115_env_cfg import PickPlace115EnvCfg
from .pick_place_116_env_cfg import PickPlace116EnvCfg
from .pick_place_117_env_cfg import PickPlace117EnvCfg
from .pick_place_118_env_cfg import PickPlace118EnvCfg
from .pick_place_119_env_cfg import PickPlace119EnvCfg
from .pick_place_120_env_cfg import PickPlace120EnvCfg
from .pick_place_121_env_cfg import PickPlace121EnvCfg
from .pick_place_122_env_cfg import PickPlace122EnvCfg
from .pick_place_123_env_cfg import PickPlace123EnvCfg
from .pick_place_124_env_cfg import PickPlace124EnvCfg
from .pick_place_125_env_cfg import PickPlace125EnvCfg
from .pick_place_126_env_cfg import PickPlace126EnvCfg
from .pick_place_127_env_cfg import PickPlace127EnvCfg
from .pick_place_128_env_cfg import PickPlace128EnvCfg
from .pick_place_129_env_cfg import PickPlace129EnvCfg
from .pick_place_130_env_cfg import PickPlace130EnvCfg
from .pick_place_131_env_cfg import PickPlace131EnvCfg
from .pick_place_132_env_cfg import PickPlace132EnvCfg
from .pick_place_133_env_cfg import PickPlace133EnvCfg
from .pick_place_134_env_cfg import PickPlace134EnvCfg
from .pick_place_135_env_cfg import PickPlace135EnvCfg
from .pick_place_136_env_cfg import PickPlace136EnvCfg
from .pick_place_137_env_cfg import PickPlace137EnvCfg
from .pick_place_138_env_cfg import PickPlace138EnvCfg
from .pick_place_139_env_cfg import PickPlace139EnvCfg
from .pick_place_140_env_cfg import PickPlace140EnvCfg
from .pick_place_141_env_cfg import PickPlace141EnvCfg
from .pick_place_142_env_cfg import PickPlace142EnvCfg
from .pick_place_143_env_cfg import PickPlace143EnvCfg
from .pick_place_144_env_cfg import PickPlace144EnvCfg
from .pick_place_145_env_cfg import PickPlace145EnvCfg
from .pick_place_146_env_cfg import PickPlace146EnvCfg
from .pick_place_147_env_cfg import PickPlace147EnvCfg
from .pick_place_148_env_cfg import PickPlace148EnvCfg
from .pick_place_149_env_cfg import PickPlace149EnvCfg
from .pick_place_150_env_cfg import PickPlace150EnvCfg
from .pick_place_151_env_cfg import PickPlace151EnvCfg
from .pick_place_152_env_cfg import PickPlace152EnvCfg
from .pick_place_153_env_cfg import PickPlace153EnvCfg
from .pick_place_154_env_cfg import PickPlace154EnvCfg
from .pick_place_155_env_cfg import PickPlace155EnvCfg
from .pick_place_156_env_cfg import PickPlace156EnvCfg
from .pick_place_157_env_cfg import PickPlace157EnvCfg
from .pick_place_158_env_cfg import PickPlace158EnvCfg
from .pick_place_159_env_cfg import PickPlace159EnvCfg
from .pick_place_160_env_cfg import PickPlace160EnvCfg
from .pick_place_161_env_cfg import PickPlace161EnvCfg
from .pick_place_162_env_cfg import PickPlace162EnvCfg
from .pick_place_163_env_cfg import PickPlace163EnvCfg
from .pick_place_164_env_cfg import PickPlace164EnvCfg
from .pick_place_165_env_cfg import PickPlace165EnvCfg
from .pick_place_166_env_cfg import PickPlace166EnvCfg
from .pick_place_167_env_cfg import PickPlace167EnvCfg
from .pick_place_168_env_cfg import PickPlace168EnvCfg
from .pick_place_169_env_cfg import PickPlace169EnvCfg
from .pick_place_170_env_cfg import PickPlace170EnvCfg
from .pick_place_171_env_cfg import PickPlace171EnvCfg
from .pick_place_172_env_cfg import PickPlace172EnvCfg
from .pick_place_173_env_cfg import PickPlace173EnvCfg
from .pick_place_174_env_cfg import PickPlace174EnvCfg
from .pick_place_175_env_cfg import PickPlace175EnvCfg
from .pick_place_176_env_cfg import PickPlace176EnvCfg
from .pick_place_177_env_cfg import PickPlace177EnvCfg
from .pick_place_178_env_cfg import PickPlace178EnvCfg
from .pick_place_179_env_cfg import PickPlace179EnvCfg
from .pick_place_180_env_cfg import PickPlace180EnvCfg
from .pick_place_181_env_cfg import PickPlace181EnvCfg
from .pick_place_182_env_cfg import PickPlace182EnvCfg
from .pick_place_183_env_cfg import PickPlace183EnvCfg
from .pick_place_184_env_cfg import PickPlace184EnvCfg
from .pick_place_185_env_cfg import PickPlace185EnvCfg
from .pick_place_186_env_cfg import PickPlace186EnvCfg
from .pick_place_187_env_cfg import PickPlace187EnvCfg
from .pick_place_188_env_cfg import PickPlace188EnvCfg
from .pick_place_189_env_cfg import PickPlace189EnvCfg
from .pick_place_190_env_cfg import PickPlace190EnvCfg
from .pick_place_191_env_cfg import PickPlace191EnvCfg
from .pick_place_192_env_cfg import PickPlace192EnvCfg
from .pick_place_193_env_cfg import PickPlace193EnvCfg
from .pick_place_194_env_cfg import PickPlace194EnvCfg
from .pick_place_195_env_cfg import PickPlace195EnvCfg
from .pick_place_196_env_cfg import PickPlace196EnvCfg
from .pick_place_197_env_cfg import PickPlace197EnvCfg
from .pick_place_198_env_cfg import PickPlace198EnvCfg
from .pick_place_199_env_cfg import PickPlace199EnvCfg
from .pick_place_200_env_cfg import PickPlace200EnvCfg
from .pick_place_201_env_cfg import PickPlace201EnvCfg
from .pick_place_202_env_cfg import PickPlace202EnvCfg
from .pick_place_203_env_cfg import PickPlace203EnvCfg
from .pick_place_204_env_cfg import PickPlace204EnvCfg
from .pick_place_205_env_cfg import PickPlace205EnvCfg
from .pick_place_206_env_cfg import PickPlace206EnvCfg
from .pick_place_207_env_cfg import PickPlace207EnvCfg
from .pick_place_208_env_cfg import PickPlace208EnvCfg
from .pick_place_209_env_cfg import PickPlace209EnvCfg
from .pick_place_210_env_cfg import PickPlace210EnvCfg
from .pick_place_211_env_cfg import PickPlace211EnvCfg
from .pick_place_212_env_cfg import PickPlace212EnvCfg
from .pick_place_213_env_cfg import PickPlace213EnvCfg
from .pick_place_214_env_cfg import PickPlace214EnvCfg
from .pick_place_215_env_cfg import PickPlace215EnvCfg
from .pick_place_216_env_cfg import PickPlace216EnvCfg
from .pick_place_217_env_cfg import PickPlace217EnvCfg
from .pick_place_218_env_cfg import PickPlace218EnvCfg
from .pick_place_219_env_cfg import PickPlace219EnvCfg
from .pick_place_220_env_cfg import PickPlace220EnvCfg
from .pick_place_221_env_cfg import PickPlace221EnvCfg
from .pick_place_222_env_cfg import PickPlace222EnvCfg
from .pick_place_223_env_cfg import PickPlace223EnvCfg
from .pick_place_224_env_cfg import PickPlace224EnvCfg
from .pick_place_225_env_cfg import PickPlace225EnvCfg
from .pick_place_226_env_cfg import PickPlace226EnvCfg
from .pick_place_227_env_cfg import PickPlace227EnvCfg
from .pick_place_228_env_cfg import PickPlace228EnvCfg
from .pick_place_229_env_cfg import PickPlace229EnvCfg
from .pick_place_230_env_cfg import PickPlace230EnvCfg
from .pick_place_231_env_cfg import PickPlace231EnvCfg
from .pick_place_232_env_cfg import PickPlace232EnvCfg
from .pick_place_233_env_cfg import PickPlace233EnvCfg
from .pick_place_234_env_cfg import PickPlace234EnvCfg
from .pick_place_235_env_cfg import PickPlace235EnvCfg
from .pick_place_236_env_cfg import PickPlace236EnvCfg
from .pick_place_237_env_cfg import PickPlace237EnvCfg
from .pick_place_238_env_cfg import PickPlace238EnvCfg
from .pick_place_239_env_cfg import PickPlace239EnvCfg
from .pick_place_240_env_cfg import PickPlace240EnvCfg
from .pick_place_241_env_cfg import PickPlace241EnvCfg
from .pick_place_242_env_cfg import PickPlace242EnvCfg
from .pick_place_243_env_cfg import PickPlace243EnvCfg
from .pick_place_244_env_cfg import PickPlace244EnvCfg
from .pick_place_245_env_cfg import PickPlace245EnvCfg
from .pick_place_246_env_cfg import PickPlace246EnvCfg
from .pick_place_247_env_cfg import PickPlace247EnvCfg
from .pick_place_248_env_cfg import PickPlace248EnvCfg
from .pick_place_249_env_cfg import PickPlace249EnvCfg
from .pick_place_250_env_cfg import PickPlace250EnvCfg
from .pick_place_251_env_cfg import PickPlace251EnvCfg
from .pick_place_252_env_cfg import PickPlace252EnvCfg
from .pick_place_253_env_cfg import PickPlace253EnvCfg
from .pick_place_254_env_cfg import PickPlace254EnvCfg
from .pick_place_255_env_cfg import PickPlace255EnvCfg


@configclass
class VLAHeterogeneousEnvCfg(VLABenchmarkEnvCfg):
    """Environments that each run one of the 256 example pick-and-place tasks on furniture, switched on reset.

    The tasks stand objects of the RoboLab and Omniverse object libraries on a piece of furniture: a table, desk,
    cabinet or shelf. Each task puts objects in or takes them out of a container, puts them on a flat object or
    stacks them; some placements require the objects upright. Select the tasks with
    :meth:`~isaaclab.managers.HeterogeneousTaskManager.set_task`; each environment follows the instruction of its
    task. The success terms of all tasks are evaluated together by
    :class:`~isaaclab_tasks.contrib.heterogeneous_tasks.mdp.task_success`.
    """

    instruction = None
    episode_length_s = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.terminations.success = DoneTerm(func=mdp.task_success)
        self.tasks = {
            "PickPlace000": PickPlace000EnvCfg(),
            "PickPlace001": PickPlace001EnvCfg(),
            "PickPlace002": PickPlace002EnvCfg(),
            "PickPlace003": PickPlace003EnvCfg(),
            "PickPlace004": PickPlace004EnvCfg(),
            "PickPlace005": PickPlace005EnvCfg(),
            "PickPlace006": PickPlace006EnvCfg(),
            "PickPlace007": PickPlace007EnvCfg(),
            "PickPlace008": PickPlace008EnvCfg(),
            "PickPlace009": PickPlace009EnvCfg(),
            "PickPlace010": PickPlace010EnvCfg(),
            "PickPlace011": PickPlace011EnvCfg(),
            "PickPlace012": PickPlace012EnvCfg(),
            "PickPlace013": PickPlace013EnvCfg(),
            "PickPlace014": PickPlace014EnvCfg(),
            "PickPlace015": PickPlace015EnvCfg(),
            "PickPlace016": PickPlace016EnvCfg(),
            "PickPlace017": PickPlace017EnvCfg(),
            "PickPlace018": PickPlace018EnvCfg(),
            "PickPlace019": PickPlace019EnvCfg(),
            "PickPlace020": PickPlace020EnvCfg(),
            "PickPlace021": PickPlace021EnvCfg(),
            "PickPlace022": PickPlace022EnvCfg(),
            "PickPlace023": PickPlace023EnvCfg(),
            "PickPlace024": PickPlace024EnvCfg(),
            "PickPlace025": PickPlace025EnvCfg(),
            "PickPlace026": PickPlace026EnvCfg(),
            "PickPlace027": PickPlace027EnvCfg(),
            "PickPlace028": PickPlace028EnvCfg(),
            "PickPlace029": PickPlace029EnvCfg(),
            "PickPlace030": PickPlace030EnvCfg(),
            "PickPlace031": PickPlace031EnvCfg(),
            "PickPlace032": PickPlace032EnvCfg(),
            "PickPlace033": PickPlace033EnvCfg(),
            "PickPlace034": PickPlace034EnvCfg(),
            "PickPlace035": PickPlace035EnvCfg(),
            "PickPlace036": PickPlace036EnvCfg(),
            "PickPlace037": PickPlace037EnvCfg(),
            "PickPlace038": PickPlace038EnvCfg(),
            "PickPlace039": PickPlace039EnvCfg(),
            "PickPlace040": PickPlace040EnvCfg(),
            "PickPlace041": PickPlace041EnvCfg(),
            "PickPlace042": PickPlace042EnvCfg(),
            "PickPlace043": PickPlace043EnvCfg(),
            "PickPlace044": PickPlace044EnvCfg(),
            "PickPlace045": PickPlace045EnvCfg(),
            "PickPlace046": PickPlace046EnvCfg(),
            "PickPlace047": PickPlace047EnvCfg(),
            "PickPlace048": PickPlace048EnvCfg(),
            "PickPlace049": PickPlace049EnvCfg(),
            "PickPlace050": PickPlace050EnvCfg(),
            "PickPlace051": PickPlace051EnvCfg(),
            "PickPlace052": PickPlace052EnvCfg(),
            "PickPlace053": PickPlace053EnvCfg(),
            "PickPlace054": PickPlace054EnvCfg(),
            "PickPlace055": PickPlace055EnvCfg(),
            "PickPlace056": PickPlace056EnvCfg(),
            "PickPlace057": PickPlace057EnvCfg(),
            "PickPlace058": PickPlace058EnvCfg(),
            "PickPlace059": PickPlace059EnvCfg(),
            "PickPlace060": PickPlace060EnvCfg(),
            "PickPlace061": PickPlace061EnvCfg(),
            "PickPlace062": PickPlace062EnvCfg(),
            "PickPlace063": PickPlace063EnvCfg(),
            "PickPlace064": PickPlace064EnvCfg(),
            "PickPlace065": PickPlace065EnvCfg(),
            "PickPlace066": PickPlace066EnvCfg(),
            "PickPlace067": PickPlace067EnvCfg(),
            "PickPlace068": PickPlace068EnvCfg(),
            "PickPlace069": PickPlace069EnvCfg(),
            "PickPlace070": PickPlace070EnvCfg(),
            "PickPlace071": PickPlace071EnvCfg(),
            "PickPlace072": PickPlace072EnvCfg(),
            "PickPlace073": PickPlace073EnvCfg(),
            "PickPlace074": PickPlace074EnvCfg(),
            "PickPlace075": PickPlace075EnvCfg(),
            "PickPlace076": PickPlace076EnvCfg(),
            "PickPlace077": PickPlace077EnvCfg(),
            "PickPlace078": PickPlace078EnvCfg(),
            "PickPlace079": PickPlace079EnvCfg(),
            "PickPlace080": PickPlace080EnvCfg(),
            "PickPlace081": PickPlace081EnvCfg(),
            "PickPlace082": PickPlace082EnvCfg(),
            "PickPlace083": PickPlace083EnvCfg(),
            "PickPlace084": PickPlace084EnvCfg(),
            "PickPlace085": PickPlace085EnvCfg(),
            "PickPlace086": PickPlace086EnvCfg(),
            "PickPlace087": PickPlace087EnvCfg(),
            "PickPlace088": PickPlace088EnvCfg(),
            "PickPlace089": PickPlace089EnvCfg(),
            "PickPlace090": PickPlace090EnvCfg(),
            "PickPlace091": PickPlace091EnvCfg(),
            "PickPlace092": PickPlace092EnvCfg(),
            "PickPlace093": PickPlace093EnvCfg(),
            "PickPlace094": PickPlace094EnvCfg(),
            "PickPlace095": PickPlace095EnvCfg(),
            "PickPlace096": PickPlace096EnvCfg(),
            "PickPlace097": PickPlace097EnvCfg(),
            "PickPlace098": PickPlace098EnvCfg(),
            "PickPlace099": PickPlace099EnvCfg(),
            "PickPlace100": PickPlace100EnvCfg(),
            "PickPlace101": PickPlace101EnvCfg(),
            "PickPlace102": PickPlace102EnvCfg(),
            "PickPlace103": PickPlace103EnvCfg(),
            "PickPlace104": PickPlace104EnvCfg(),
            "PickPlace105": PickPlace105EnvCfg(),
            "PickPlace106": PickPlace106EnvCfg(),
            "PickPlace107": PickPlace107EnvCfg(),
            "PickPlace108": PickPlace108EnvCfg(),
            "PickPlace109": PickPlace109EnvCfg(),
            "PickPlace110": PickPlace110EnvCfg(),
            "PickPlace111": PickPlace111EnvCfg(),
            "PickPlace112": PickPlace112EnvCfg(),
            "PickPlace113": PickPlace113EnvCfg(),
            "PickPlace114": PickPlace114EnvCfg(),
            "PickPlace115": PickPlace115EnvCfg(),
            "PickPlace116": PickPlace116EnvCfg(),
            "PickPlace117": PickPlace117EnvCfg(),
            "PickPlace118": PickPlace118EnvCfg(),
            "PickPlace119": PickPlace119EnvCfg(),
            "PickPlace120": PickPlace120EnvCfg(),
            "PickPlace121": PickPlace121EnvCfg(),
            "PickPlace122": PickPlace122EnvCfg(),
            "PickPlace123": PickPlace123EnvCfg(),
            "PickPlace124": PickPlace124EnvCfg(),
            "PickPlace125": PickPlace125EnvCfg(),
            "PickPlace126": PickPlace126EnvCfg(),
            "PickPlace127": PickPlace127EnvCfg(),
            "PickPlace128": PickPlace128EnvCfg(),
            "PickPlace129": PickPlace129EnvCfg(),
            "PickPlace130": PickPlace130EnvCfg(),
            "PickPlace131": PickPlace131EnvCfg(),
            "PickPlace132": PickPlace132EnvCfg(),
            "PickPlace133": PickPlace133EnvCfg(),
            "PickPlace134": PickPlace134EnvCfg(),
            "PickPlace135": PickPlace135EnvCfg(),
            "PickPlace136": PickPlace136EnvCfg(),
            "PickPlace137": PickPlace137EnvCfg(),
            "PickPlace138": PickPlace138EnvCfg(),
            "PickPlace139": PickPlace139EnvCfg(),
            "PickPlace140": PickPlace140EnvCfg(),
            "PickPlace141": PickPlace141EnvCfg(),
            "PickPlace142": PickPlace142EnvCfg(),
            "PickPlace143": PickPlace143EnvCfg(),
            "PickPlace144": PickPlace144EnvCfg(),
            "PickPlace145": PickPlace145EnvCfg(),
            "PickPlace146": PickPlace146EnvCfg(),
            "PickPlace147": PickPlace147EnvCfg(),
            "PickPlace148": PickPlace148EnvCfg(),
            "PickPlace149": PickPlace149EnvCfg(),
            "PickPlace150": PickPlace150EnvCfg(),
            "PickPlace151": PickPlace151EnvCfg(),
            "PickPlace152": PickPlace152EnvCfg(),
            "PickPlace153": PickPlace153EnvCfg(),
            "PickPlace154": PickPlace154EnvCfg(),
            "PickPlace155": PickPlace155EnvCfg(),
            "PickPlace156": PickPlace156EnvCfg(),
            "PickPlace157": PickPlace157EnvCfg(),
            "PickPlace158": PickPlace158EnvCfg(),
            "PickPlace159": PickPlace159EnvCfg(),
            "PickPlace160": PickPlace160EnvCfg(),
            "PickPlace161": PickPlace161EnvCfg(),
            "PickPlace162": PickPlace162EnvCfg(),
            "PickPlace163": PickPlace163EnvCfg(),
            "PickPlace164": PickPlace164EnvCfg(),
            "PickPlace165": PickPlace165EnvCfg(),
            "PickPlace166": PickPlace166EnvCfg(),
            "PickPlace167": PickPlace167EnvCfg(),
            "PickPlace168": PickPlace168EnvCfg(),
            "PickPlace169": PickPlace169EnvCfg(),
            "PickPlace170": PickPlace170EnvCfg(),
            "PickPlace171": PickPlace171EnvCfg(),
            "PickPlace172": PickPlace172EnvCfg(),
            "PickPlace173": PickPlace173EnvCfg(),
            "PickPlace174": PickPlace174EnvCfg(),
            "PickPlace175": PickPlace175EnvCfg(),
            "PickPlace176": PickPlace176EnvCfg(),
            "PickPlace177": PickPlace177EnvCfg(),
            "PickPlace178": PickPlace178EnvCfg(),
            "PickPlace179": PickPlace179EnvCfg(),
            "PickPlace180": PickPlace180EnvCfg(),
            "PickPlace181": PickPlace181EnvCfg(),
            "PickPlace182": PickPlace182EnvCfg(),
            "PickPlace183": PickPlace183EnvCfg(),
            "PickPlace184": PickPlace184EnvCfg(),
            "PickPlace185": PickPlace185EnvCfg(),
            "PickPlace186": PickPlace186EnvCfg(),
            "PickPlace187": PickPlace187EnvCfg(),
            "PickPlace188": PickPlace188EnvCfg(),
            "PickPlace189": PickPlace189EnvCfg(),
            "PickPlace190": PickPlace190EnvCfg(),
            "PickPlace191": PickPlace191EnvCfg(),
            "PickPlace192": PickPlace192EnvCfg(),
            "PickPlace193": PickPlace193EnvCfg(),
            "PickPlace194": PickPlace194EnvCfg(),
            "PickPlace195": PickPlace195EnvCfg(),
            "PickPlace196": PickPlace196EnvCfg(),
            "PickPlace197": PickPlace197EnvCfg(),
            "PickPlace198": PickPlace198EnvCfg(),
            "PickPlace199": PickPlace199EnvCfg(),
            "PickPlace200": PickPlace200EnvCfg(),
            "PickPlace201": PickPlace201EnvCfg(),
            "PickPlace202": PickPlace202EnvCfg(),
            "PickPlace203": PickPlace203EnvCfg(),
            "PickPlace204": PickPlace204EnvCfg(),
            "PickPlace205": PickPlace205EnvCfg(),
            "PickPlace206": PickPlace206EnvCfg(),
            "PickPlace207": PickPlace207EnvCfg(),
            "PickPlace208": PickPlace208EnvCfg(),
            "PickPlace209": PickPlace209EnvCfg(),
            "PickPlace210": PickPlace210EnvCfg(),
            "PickPlace211": PickPlace211EnvCfg(),
            "PickPlace212": PickPlace212EnvCfg(),
            "PickPlace213": PickPlace213EnvCfg(),
            "PickPlace214": PickPlace214EnvCfg(),
            "PickPlace215": PickPlace215EnvCfg(),
            "PickPlace216": PickPlace216EnvCfg(),
            "PickPlace217": PickPlace217EnvCfg(),
            "PickPlace218": PickPlace218EnvCfg(),
            "PickPlace219": PickPlace219EnvCfg(),
            "PickPlace220": PickPlace220EnvCfg(),
            "PickPlace221": PickPlace221EnvCfg(),
            "PickPlace222": PickPlace222EnvCfg(),
            "PickPlace223": PickPlace223EnvCfg(),
            "PickPlace224": PickPlace224EnvCfg(),
            "PickPlace225": PickPlace225EnvCfg(),
            "PickPlace226": PickPlace226EnvCfg(),
            "PickPlace227": PickPlace227EnvCfg(),
            "PickPlace228": PickPlace228EnvCfg(),
            "PickPlace229": PickPlace229EnvCfg(),
            "PickPlace230": PickPlace230EnvCfg(),
            "PickPlace231": PickPlace231EnvCfg(),
            "PickPlace232": PickPlace232EnvCfg(),
            "PickPlace233": PickPlace233EnvCfg(),
            "PickPlace234": PickPlace234EnvCfg(),
            "PickPlace235": PickPlace235EnvCfg(),
            "PickPlace236": PickPlace236EnvCfg(),
            "PickPlace237": PickPlace237EnvCfg(),
            "PickPlace238": PickPlace238EnvCfg(),
            "PickPlace239": PickPlace239EnvCfg(),
            "PickPlace240": PickPlace240EnvCfg(),
            "PickPlace241": PickPlace241EnvCfg(),
            "PickPlace242": PickPlace242EnvCfg(),
            "PickPlace243": PickPlace243EnvCfg(),
            "PickPlace244": PickPlace244EnvCfg(),
            "PickPlace245": PickPlace245EnvCfg(),
            "PickPlace246": PickPlace246EnvCfg(),
            "PickPlace247": PickPlace247EnvCfg(),
            "PickPlace248": PickPlace248EnvCfg(),
            "PickPlace249": PickPlace249EnvCfg(),
            "PickPlace250": PickPlace250EnvCfg(),
            "PickPlace251": PickPlace251EnvCfg(),
            "PickPlace252": PickPlace252EnvCfg(),
            "PickPlace253": PickPlace253EnvCfg(),
            "PickPlace254": PickPlace254EnvCfg(),
            "PickPlace255": PickPlace255EnvCfg(),
        }
