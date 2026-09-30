from suqc.parameter.postchanges import PostScenarioChange


class ChangeInitialPathogenLoad(PostScenarioChange):
    def __init__(self, init_pathogen_load):
        super(ChangeInitialPathogenLoad, self).__init__(name="initial_pathogen_load")
        self.init_pathogen_load = init_pathogen_load

    def get_changes_dict(self, scenario, parameter_id, run_id, parameter_variation):
        return {"initialPathogenLoad": self.init_pathogen_load}
