# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy

class DextrahADR():
    def __init__(self, event_manager, adr_cfg_dict, adr_custom_cfg_dict):
        self.event_manager = event_manager
        self.adr_cfg_dict = adr_cfg_dict
        self.adr_custom_cfg_dict = adr_custom_cfg_dict

        # Copy the event manager so we can retain the original parameter
        # ranges
        self.adr_cfg_dict_initial = copy.deepcopy(adr_cfg_dict)

        # Save initial parameter ranges
        self.save_param_ranges()

        # Increment counter
        self.increment_counter = 0

    def save_param_ranges(self):
        for term_name, term_params in self.adr_cfg_dict.items():
            if term_name != "num_increments":
                term = self.event_manager.get_term_cfg(term_name)
                for param_name, param_values in term_params.items():
                    self.adr_cfg_dict_initial[term_name][param_name] =\
                        copy.deepcopy(term.params[param_name])

    def print_params(self):
        for term_name, term_params in self.adr_cfg_dict.items():
            if term_name != "num_increments":
                term = self.event_manager.get_term_cfg(term_name)
                print('term_name', term)

    def increase_ranges(self, increase_counter=True):
        # First check if you've met or exceeded the number of incremental changes
        # allowed
        if self.increment_counter >= self.adr_cfg_dict["num_increments"]:
            self.increment_counter = self.adr_cfg_dict["num_increments"]
            print('---------Maximum ADR hit-----------')
        else:
            if increase_counter:
                print('Incrementing ADR')
                self.increment_counter += 1

        for term_name, term_params in self.adr_cfg_dict.items():
            if term_name != "num_increments":
                term = self.event_manager.get_term_cfg(term_name)
                for param_name, param_values in term_params.items():
                    default_val = self.adr_cfg_dict_initial[term_name][param_name][0]
                    val_low = self.adr_cfg_dict[term_name][param_name][0]
                    val_high = self.adr_cfg_dict[term_name][param_name][1]

                    lower_limit_inc = (val_low - default_val) / float(self.adr_cfg_dict["num_increments"])
                    lower_limit = lower_limit_inc * self.increment_counter + default_val

                    upper_limit_inc =(val_high - default_val) / float(self.adr_cfg_dict["num_increments"])
                    upper_limit = upper_limit_inc * self.increment_counter + self.adr_cfg_dict_initial[term_name][param_name][1]
                    new_range = (lower_limit, upper_limit)
                    term.params[param_name] = new_range

    def num_increments(self):
        return self.increment_counter

    def set_num_increments(self, num_increments):
        self.increment_counter = num_increments
        self.increase_ranges(increase_counter=False)

    def get_custom_param_value(self, param_group, param_name):

        upper_limit = self.adr_custom_cfg_dict[param_group][param_name][1]
        lower_limit = self.adr_custom_cfg_dict[param_group][param_name][0]

        param_slope = (upper_limit - lower_limit) / float(self.adr_cfg_dict["num_increments"])

        param_value = param_slope * self.increment_counter + lower_limit

        return param_value
