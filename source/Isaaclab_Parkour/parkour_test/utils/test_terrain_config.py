from parkour_isaaclab.terrains.extreme_parkour.extreme_parkour_terrains_cfg import *  
from parkour_isaaclab.terrains.parkour_terrain_generator_cfg import ParkourTerrainGeneratorCfg

PARKOUR_TERRAINS_CFG = ParkourTerrainGeneratorCfg(
    size=(22.0, 12.0),
    border_width=5.0,
    num_rows=2,
    num_cols=2,
    # horizontal_scale=0.05,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=1.5,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    num_goals = 8,
    curriculum= True,
    sub_terrains={

        # "parkour_gap": ExtremeParkourGapTerrainCfg(
        #                 proportion=0.2,
        #                 apply_roughness=True,
        #                 x_range = (0.8, 1.5),
        #                 y_range = (-0.1, 0.1),
        #                 half_valid_width = (0.6, 1.2),
        #                 )
        # "parkour_hurdle": ExtremeParkourHurdleTerrainCfg(
        #                 proportion=0.2,
        #                 apply_roughness=True,
        #                 x_range = (1.2, 2.2),
        #                 y_range = (-0.1, 0.1),
        #                 half_valid_width = (0.4,0.8),
        #                 hurdle_height_range= '0.1+0.1*difficulty, 0.15+0.25*difficulty'
        #                 ),

        # "parkour_flat": ExtremeParkourHurdleTerrainCfg(
        #                 proportion=0.2,
        #                 apply_roughness=True,
        #                 apply_flat=True,
        #                 x_range = (1.2, 2.2),
        #                 y_range = (-0.1, 0.1),
        #                 half_valid_width = (0.4,0.8),
        #                 hurdle_height_range= '0.1+0.1*difficulty, 0.15+0.15*difficulty'
        #                 ),

        # "parkour_step": ExtremeParkourStepTerrainCfg(
        #                 proportion=0.2,
        #                 apply_roughness=True,
        #                 x_range = (0.3,1.5),
        #                 y_range = (-0.1, 0.1),
        #                 half_valid_width = (0.5, 1)
        #                 ),

        # "parkour": ExtremeParkourTerrainCfg(
        #                 proportion=0.2,
        #                 apply_roughness=True,
        #                 y_range = (-0.1, 0.1),


        #                 ),
        # "parkour_demo": ExtremeParkourDemoTerrainCfg(
        #         y_range = (-0.1, 0.1),
        #         proportion=0.0,
        #         apply_roughness=True,
        #         ),



    },
)