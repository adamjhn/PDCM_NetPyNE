import bluepyopt.ephys as ephys
from bluepyopt.ephys.models import CellModel
from bluepyopt.parameters import Parameter
from neuron import rxd

try:
    basestring
except NameError:
    basestring = str


class RxDCellModel(CellModel):

    """Cell model class with added rxd"""

    def __init__(
        self,
        name,
        morph=None,
        mechs=None,
        params=None,
        gid=0,
        regions=None,
        species=None,
        states=None,
        parameters=None,
        reactions=None,
        multiCompartmentReactions=None,
        rates=None,
        constants=None,
    ):

        """Constructor

        Args:
            name (str): name of this object
                        should be alphanumeric string, underscores are allowed,
                        first char should be a letter
            morph (Morphology):
                underlying Morphology of the cell
            mechs (list of Mechanisms):
                Mechanisms associated with the cell
            params (list of Parameters):
                Parameters of the cell model
            seclist_names (list of strings):
                Names of the lists of sections
            secarray_names (list of strings):
                Names of the sections
            regions (dict):
                RxD regions
            species (dict):
                RxD species
            reactions (dict):
                RxD rates, reactions and MultiCompartmentReactions
        """

        # Create the model
        super(RxDCellModel, self).__init__(
            name, morph=morph, mechs=mechs, params=params, gid=gid
        )
        # Store the rxd part of the model
        self.rxdRegions = regions
        self.rxdSpecies = species
        self.rxdStates = states
        self.rxdParameters = parameters
        self.rxdReactions = reactions
        self.rxdMultiCompartmentReactions = multiCompartmentReactions
        self.rxdRates = rates
        self.rxdConstants = constants
        self._rxdRegions = {}
        self._rxdSpecies = {}
        self._rxdStates = {}
        self._rxdReactions = {}
        self._rxdRates = {}

    def instantiate(self, sim=None):
        super(RxDCellModel, self).instantiate(sim)

        # Parse dicts to rxd models based on
        # https://github.com/suny-downstate-medical-center/netpyne/blob/development
        # netpyne/network/netrxd.py

        if self.rxdRegions is not None:
            self.addRegions()
        if self.rxdSpecies is not None:
            self.addSpecies()
        if self.rxdStates is not None:
            self.addStates()
        if self.rxdParameters is not None:
            self.addParameters()
        if self.rxdReactions is not None:
            self.addReactions(multicompartment=False)
        if self.rxdMultiCompartmentReactions is not None:
            self.addReactions(multicompartment=True)
        if self.rxdRates is not None:
            self.addRates()

        for param in self.params.values():
            if hasattr(param, "doInstantiate"):
                param.doInstantiate(self)

    def _replaceRxDStr(
        self, origStr, constants=True, regions=True, species=True, parameters=True
    ):
        # -----------------------------------------------------------------------------
        # Replace RxD param strings with expression
        # -----------------------------------------------------------------------------

        import re

        replacedStr = str(origStr)

        mapping = {}

        # replace constants
        if constants:
            constantsList = [
                c for c in self.rxdConstants if c in origStr
            ]  # get list of variables used (eg. post_ynorm or dist_xyz)
            for constantLabel in constantsList:
                mapping[constantLabel] = 'self.rxdConstants["%s"]' % (constantLabel)

        # replace regions
        if regions and hasattr(self, "_rxdRegions"):
            for regionLabel in self._rxdRegions:
                mapping[regionLabel] = 'self._rxdRegions["%s"]' % (regionLabel)

        # replace species
        if species and hasattr(self, "_rxdSpecies"):
            for speciesLabel in self._rxdSpecies:
                mapping[speciesLabel] = 'self._rxdSpecies["%s"]' % (speciesLabel)

        if species and hasattr(self, "_rxdStates"):
            for statesLabel in self._rxdStates:
                mapping[statesLabel] = 'self._rxdStates["%s"]' % (statesLabel)

        if parameters and hasattr(self, "_rxdParameters"):
            for paramLabel in self._rxdParameters:
                mapping[paramLabel] = 'self._rxdParameters["%s"]' % (paramLabel)

        # Place longer ones first to keep shorter substrings from matching where the longer ones should take place
        substrs = sorted(mapping, key=len, reverse=True)

        # Create a big OR regex that matches any of the substrings to replace
        regexp = re.compile("|".join(map(re.escape, substrs)))

        # For each match, look up the new string in the mapping
        replacedStr = regexp.sub(lambda match: mapping[match.group(0)], replacedStr)

        return replacedStr

    def addRegions(self):
        # -----------------------------------------------------------------------------
        # Add RxD regions
        # -----------------------------------------------------------------------------
        for label, param in self.rxdRegions.items():

            if "extracellular" in param and param["extracellular"] == True:
                self._rxdRegions[label] = rxd.Extracellular(
                    **{k: v for k, v in param.items() if k != "extracellular"}
                )
                continue
            # secs
            if "secs" not in param:
                param["secs"] = ["all"]
            if not isinstance(param["secs"], list):
                param["secs"] = [param["secs"]]

            # nrn_region
            if "nrn_region" not in param:
                param["nrn_region"] = None

            # geomery
            if "geometry" not in param:
                param["geometry"] = None
            geometry = param["geometry"]
            if isinstance(geometry, dict):
                try:
                    if "args" in geometry:
                        geometry = getattr(rxd, param["geometry"]["class"])(
                            **param["geometry"]["args"]
                        )
                except:
                    print(
                        "  Error creating %s Region geometry using %s class"
                        % (label, param["geometry"]["class"])
                    )
            elif isinstance(param["geometry"], str):
                geometry = getattr(rxd, param["geometry"])()

            # geomery
            if "dimension" not in param:
                param["dimension"] = None

            # geomery
            if "dx" not in param:
                param["dx"] = None

            # TODO: Can SectionList can be used in the rxd constructor
            if "all" in param["secs"]:
                if "all" in self.seclist_names:
                    nrnSecs = [sec for sec in self.icell.all]
            else:
                nrnSecs = []
                for secName in param["secs"]:
                    nrnSecs.extend([sec for sec in getattr(self.icell, secName)])

            self._rxdRegions[label] = rxd.Region(
                secs=nrnSecs,
                nrn_region=param["nrn_region"],
                geometry=geometry,
                dimension=param["dimension"],
                dx=param["dx"],
                name=label,
            )

    def _parseInitStr(
        self, label, initial, species=False, state=False, parameter=False
    ):
        funcStr = self._replaceRxDStr(
            initial, constants=True, regions=True, species=False if species else True
        )

        # create final function dynamically from string
        importStr = " from neuron import rxd"
        if species:
            afterDefStr = 'self.rxdSpecies["%s"]["initialFunc"] = initial' % (label)
        elif state:
            afterDefStr = 'self.rxdStates["%s"]["initialFunc"] = initial' % (label)
        elif parameter:
            afterDefStr = 'self.rxdParameter["%s"]["valueFunc"] = initial' % (label)
        funcStr = "def initial (node): \n%s \n return %s \n%s" % (
            importStr,
            funcStr,
            afterDefStr,
        )  # convert to lambda function
        try:
            exec(funcStr, {"rxd": rxd, "self": self})
            if species:
                initial = self.rxdSpecies[label]["initialFunc"]
            elif state:
                initial = self.rxdStates[label]["initialFunc"]
            elif parameter:
                initial = self.rxdParameter[label]["valueFunc"]
        except:
            rxdType = "Species" if species else ("State" if state else "Parameter")
            print(
                '  Error creating %s %s: cannot evaluate "initial" expression -- "%s"'
                % (rxdType, label, param["initial"])
            )
        return initial

    def addSpecies(self):

        # -----------------------------------------------------------------------------
        # Add RxD species
        # -----------------------------------------------------------------------------
        for label, param in self.rxdSpecies.items():
            # regions
            if "regions" not in param:
                print(
                    '  Error creating Species %s: "regions" parameter was missing'
                    % (label)
                )
                continue
            if not isinstance(param["regions"], list):
                param["regions"] = [param["regions"]]
            try:
                nrnRegions = [self._rxdRegions[region] for region in param["regions"]]
            except:
                print(
                    "  Error creating Species %s: could not find regions %s"
                    % (label, param["regions"])
                )

            # d
            if "d" not in param:
                param["d"] = 0

            # charge
            if "charge" not in param:
                param["charge"] = 0

            # initial
            if "initial" not in param:
                param["initial"] = None
            elif isinstance(param["initial"], basestring):
                initial = self._parseInitStr(
                    label=label, initial=param["initial"], species=True
                )

            else:
                initial = param["initial"]

            # ecs boundary condition
            if "ecs_boundary_conditions" not in param:
                param["ecs_boundary_conditions"] = None

            # atolscale
            if "atolscale" not in param:
                param["atolscale"] = 1

            if "name" not in param:
                name = label
            else:
                name = param["name"]

            # call rxd method to create Species
            self._rxdSpecies[name] = rxd.Species(
                regions=nrnRegions,
                d=param["d"],
                charge=param["charge"],
                initial=initial,
                atolscale=param["atolscale"],
                name=name,
                ecs_boundary_conditions=param["ecs_boundary_conditions"],
            )

    def addStates(self):

        # -----------------------------------------------------------------------------
        # Add RxD state
        # -----------------------------------------------------------------------------
        for label, param in self.rxdStates.items():
            # regions
            if "regions" not in param:
                print(
                    '  Error creating Species %s: "regions" parameter was missing'
                    % (label)
                )
                continue
            if not isinstance(param["regions"], list):
                param["regions"] = [param["regions"]]
            try:
                nrnRegions = [self._rxdRegions[region] for region in param["regions"]]
            except:
                print(
                    "  Error creating Species %s: could not find regions %s"
                    % (label, param["regions"])
                )

            # charge
            if "charge" not in param:
                param["charge"] = 0

            # initial
            if "initial" not in param:
                param["initial"] = None
            elif isinstance(param["initial"], basestring):
                initial = self._parseInitStr(
                    label=label, initial=param["initial"], state=True
                )

            else:
                initial = param["initial"]

            # atolscale
            if "atolscale" not in param:
                param["atolscale"] = 1

            if "name" not in param:
                name = label
            else:
                name = param["name"]

            # call rxd method to create State
            self._rxdStates[name] = rxd.State(
                regions=nrnRegions,
                charge=param["charge"],
                initial=initial,
                atolscale=param["atolscale"],
                name=name,
            )

    def addParameter(self):

        # -----------------------------------------------------------------------------
        # Add RxD parameter
        # -----------------------------------------------------------------------------
        for label, param in self.parameters.items():
            # regions
            if "regions" not in param:
                print(
                    '  Error creating Species %s: "regions" parameter was missing'
                    % (label)
                )
                continue
            if not isinstance(param["regions"], list):
                param["regions"] = [param["regions"]]
            try:
                nrnRegions = [self.__rxdRegions[region] for region in param["regions"]]
            except:
                print(
                    "  Error creating Species %s: could not find regions %s"
                    % (label, param["regions"])
                )

            # initial
            if "value" not in param:
                value = None
            elif isinstance(param["value"], basestring):
                value = self._parseInitStr(
                    label=label, initial=param["value"], parameter=True
                )
            else:
                value = param["value"]

            # atolscale
            if "atolscale" not in param:
                param["atolscale"] = 1

            if "name" not in param:
                name = label
            else:
                name = param["name"]

            # call rxd method to create Parameter
            self._rxdParameter[name] = rxd.Parameter(
                regions=nrnRegions,
                charge=param["charge"],
                value=value,
                atolscale=param["atolscale"],
                name=name,
            )

    def addReactions(self, multicompartment=False):
        # -----------------------------------------------------------------------------
        # Add RxD reactions
        # -----------------------------------------------------------------------------

        reactionStr = "MultiCompartmentReaction" if multicompartment else "Reaction"
        reactionDictKey = (
            "multicompartmentReactions" if multicompartment else "reactions"
        )
        reactDict = (
            self.rxdMultiCompartmentReactions if multicompartment else self.rxdReactions
        )

        for label, param in reactDict.items():

            dynamicVars = {"rxdmath": rxd.rxdmath, "rxd": rxd, "self": self}

            # reactant
            if "reactant" not in param:
                print(
                    '  Error creating %s %s: "reactant" parameter was missing'
                    % (reactionStr, label)
                )
                continue
            reactantStr = self._replaceRxDStr(param["reactant"])
            try:
                exec("reactant = " + reactantStr, dynamicVars)
            except TypeError:
                continue
            if "reactant" not in dynamicVars:
                dynamicVars["reactant"]  # fix for python 2

            # product
            if "product" not in param:
                print(
                    '  Error creating %s %s: "product" parameter was missing'
                    % (reactionStr, label)
                )
                continue
            productStr = self._replaceRxDStr(param["product"])

            exec("product = " + productStr, dynamicVars)
            if "product" not in dynamicVars:
                dynamicVars["product"]  # fix for python 2

            # rate_f
            if "rate_f" not in param:
                print(
                    '  Error creating %s %s: "scheme" parameter was missing'
                    % (reactionStr, label)
                )
                continue
            if isinstance(param["rate_f"], basestring):
                rate_fStr = self._replaceRxDStr(param["rate_f"])
                exec("rate_f = " + rate_fStr, dynamicVars)
                if "rate_f" not in dynamicVars:
                    dynamicVars["rate_f"]  # fix for python 2
            else:
                rate_f = param["rate_f"]

            # rate_b
            rate_b = None
            if "rate_b" not in param:
                param["rate_b"] = None
            if isinstance(param["rate_b"], basestring):
                rate_bStr = self._replaceRxDStr(param["rate_b"])
                exec("rate_b = " + rate_bStr, dynamicVars)
                if "rate_b" not in dynamicVars:
                    dynamicVars["rate_b"]  # fix for python 2
            else:
                rate_b = param["rate_b"]

            # regions
            if "regions" not in param:
                param["regions"] = [None]
            elif not isinstance(param["regions"], list):
                param["regions"] = [param["regions"]]
            try:
                nrnRegions = [
                    self._rxdRegions[region]
                    for region in param["regions"]
                    if region is not None and self._rxdRegions[region] != None
                ]
            except:
                print(
                    "  Error creating %s %s: could not find regions %s"
                    % (reactionStr, label, param["regions"])
                )

            # membrane
            if "membrane" not in param:
                param["membrane"] = None
            if param["membrane"] in self._rxdRegions:
                nrnMembraneRegion = self._rxdRegions[param["membrane"]]
            else:
                nrnMembraneRegion = None

            # custom_dynamics
            if "custom_dynamics" not in param:
                param["custom_dynamics"] = False
            if "membrane_flux" not in param:
                param["membrane_flux"] = False

            # membrane_flux
            if "membrane_flux" not in param:
                param["membrane_flux"] = False

            # scale by area
            if "scale_by_area" not in param:
                param["scale_by_area"] = True

            if rate_b is None and dynamicVars.get("rate_b", None) is None:
                # omit positional argument 'rate_b'
                self._rxdReactions[label] = getattr(rxd, reactionStr)(
                    dynamicVars["reactant"],
                    dynamicVars["product"],
                    dynamicVars["rate_f"] if "rate_f" in dynamicVars else rate_f,
                    regions=nrnRegions,
                    custom_dynamics=param["custom_dynamics"],
                    membrane_flux=param["membrane_flux"],
                    scale_by_area=param["scale_by_area"],
                    membrane=nrnMembraneRegion,
                )

            else:
                # include positional argument 'rate_b'
                self._rxdReactions[label] = getattr(rxd, reactionStr)(
                    dynamicVars["reactant"],
                    dynamicVars["product"],
                    dynamicVars["rate_f"] if "rate_f" in dynamicVars else rate_f,
                    dynamicVars["rate_b"] if "rate_b" in dynamicVars else rate_b,
                    regions=nrnRegions,
                    custom_dynamics=param["custom_dynamics"],
                    membrane_flux=param["membrane_flux"],
                    scale_by_area=param["scale_by_area"],
                    membrane=nrnMembraneRegion,
                )

    def addRates(self):
        # -----------------------------------------------------------------------------
        # Add RxD reactions
        # -----------------------------------------------------------------------------

        for label, param in self.rxdRates.items():

            dynamicVars = {"rxdmath": rxd.rxdmath, "rxd": rxd, "self": self}

            # species
            if "species" not in param:
                print(
                    '  Error creating Rate %s: "species" parameter was missing'
                    % (label)
                )
                continue
            if isinstance(param["species"], basestring):
                speciesStr = self._replaceRxDStr(param["species"])
                exec("species = " + speciesStr, dynamicVars)
                if "species" not in dynamicVars:
                    dynamicVars["species"]  # fix for python 2
            else:
                print(
                    '  Error creating Rate %s: "species" parameter should be a string'
                    % (label)
                )
                continue

            # rate
            if "rate" not in param:
                print(
                    '  Error creating Rate %s: "rate" parameter was missing' % (label)
                )
                continue
            if isinstance(param["rate"], basestring):
                rateStr = self._replaceRxDStr(param["rate"])
                exec("rate = " + rateStr, dynamicVars)
                if "rate" not in dynamicVars:
                    dynamicVars["rate"]  # fix for python 2

            # regions
            if "regions" not in param:
                param["regions"] = None
            elif not isinstance(param["regions"], list):
                param["regions"] = [param["regions"]]
            try:
                nrnRegions = [
                    self._rxdRegions[region]
                    for region in param["regions"]
                    if region is not None and self._rxdRegions[region] != None
                ]
            except:
                print(
                    "  Error creating Rate %s: could not find regions %s"
                    % (label, param["regions"])
                )

            # membrane_flux
            if "membrane_flux" not in param:
                param["membrane_flux"] = False

            self._rxdRates[label] = rxd.Rate(
                dynamicVars["species"],
                dynamicVars["rate"],
                regions=nrnRegions,
                membrane_flux=param["membrane_flux"],
            )


class RxDReactionParameter(Parameter):
    def __init__(
        self,
        name,
        value=None,
        frozen=False,
        bounds=None,
        reaction_name=None,
        regions=None,
        rate_string=None,
        rate_f=False,
        rate_b=False,
        multicompartment=False,
        param_dependencies=None,
    ):
        super(RxDReactionParameter, self).__init__(
            name, frozen=frozen, bounds=bounds, param_dependencies=param_dependencies
        )
        self.reaction_name = reaction_name
        self.regions = regions
        self.rate_f = rate_f
        self.rate_b = rate_b
        self.rate_string = rate_string
        self.multicompartment = multicompartment

    def instantiate(self, sim=None, icell=None, params=None):
        self.icell = icell
        self.params = params
        print(self, params)
        # rxd parameters are set _after_ the rxd species are created
        if hasattr(self, "cell_model") and self.cell_model is not None:
            doInstantiate(self.cell_model)

    def doInstantiate(self, cell_model=None):
        self.cell_model = cell_model
        if self.value is None:
            raise Exception(
                'RxDReactionParameter: impossible to instantiate parameter "%s" '
                "without value" % self.name
            )

        if self.reaction_name in cell_model._rxdReactions:
            react = cell_model._rxdReactions[self.reaction_name]
        else:
            raise Exception(
                'RxDReactionParameter: impossible to instantiate parameter "%s" '
                'reaction_name "%s" not found on cell' % (self.name, self.reaction_name)
            )
        if self.rate_string is not None:
            rate = self._replaceRxDStr(self.rate_string % self.value)
        else:
            rate = self.value
        if self.rate_f:
            react.rate_f = rate
        if self.rate_b:
            react.rate_b = rate

    def destroy(self, sim=None):
        """Remove parameter from the simulator"""
        pass


class RxDSpeciesParameter(Parameter):
    def __init__(
        self,
        name,
        value=None,
        frozen=False,
        bounds=None,
        species_name=None,
        regions=None,
        species_string=None,
        initial_param=False,
        diffusion_param=False,
        param_dependencies=None,
    ):
        super(RxDSpeciesParameter, self).__init__(
            name, frozen=frozen, bounds=bounds, param_dependencies=param_dependencies
        )
        self.species_name = species_name
        self.regions = regions
        self.species_string = species_string
        self.initial_param = initial_param
        self.diffusion_param = diffusion_param

    def instantiate(self, sim=None, icell=None, params=None):
        self.icell = icell
        self.params = params

        # rxd parameters are set _after_ the rxd species are created
        if hasattr(self, "cell_model") and self.cell_model is not None:
            doInstantiate(self.cell_model)

    def doInstantiate(self, cell_model=None):
        self.cell_model = cell_model
        if self.value is None:
            raise Exception(
                'RxDSpeciesParameter: impossible to instantiate parameter "%s" '
                "without value" % self.name
            )

        """
            for sec in icell.all:
            psec = sec.psection()
            for sp in psec["species"]:
                if sp.name == species_name:
                    break
            else:
                continue
            break
        else:
        """
        if self.species_name in cell_model._rxdSpecies:
            sp = cell_model._rxdSpecies[self.species_name]
        else:
            raise Exception(
                'RxDSpeciesParameter: impossible to instantiate parameter "%s" '
                'species_name "%s" not found on cell' % (self.name, self.species_name)
            )

        if self.diffusion_param:
            if self.species_string is not None:
                sp.d = self._replaceRxDStr(self.species_string % self.value)
            else:
                sp.d = self.value

        if self.initial_param:
            if self.species_string is not None:
                sp.initial = self._parseInitStr(
                    label=label, initial=self.species_string % self.value, species=True
                )
            else:
                sp.initial = self.value

    def destroy(self, sim=None):
        """Remove parameter from the simulator"""
        pass


class RxDParameterParameter(Parameter):
    def __init__(
        self,
        name,
        value=None,
        frozen=False,
        bounds=None,
        parameter_name=None,
        parameter_string=None,
        regions=None,
        param_dependencies=None,
    ):
        super(RxDSpeciesParameter, self).__init__(
            name, frozen=frozen, bounds=bounds, param_dependencies=param_dependencies
        )
        self.parameter_name = species_name
        self.regions = regions
        self.parameter_string = parameter_string

    def instantiate(self, sim=None, icell=None, params=None):
        self.icell = icell
        self.params = params

        # rxd parameters are set _after_ the rxd species are created
        if hasattr(self, "cell_model") and self.cell_model is not None:
            doInstantiate(self.cell_model)

    def doInstantiate(self, cell_model=None):
        self.cell_model = cell_model
        if self.value is None:
            raise Exception(
                'RxDParameterParameter: impossible to instantiate parameter "%s" '
                "without value" % self.name
            )
        if self.parameter_name in cell_model._rxdParameter:
            param = cell_model._rxdParameter[self.parameter_name]
        else:
            raise Exception(
                'RxDSpeciesParameter: impossible to instantiate parameter "%s" '
                'species_name "%s" not found on cell' % (self.name, self.species_name)
            )

        if self.parameter_string is not None:
            param.value = self._parseInitStr(
                label=label, initial=self.parameter_string % self.value, parameter=True
            )
        else:
            param.value = self.value

    def destroy(self, sim=None):
        """Remove parameter from the simulator"""
        pass
