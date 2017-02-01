import enum
from .sheets import IceSheet


class RignotBasin(enum.Enum):
    @classmethod
    def parse(cls, value: str) -> "RignotBasin":
        value = value.upper()
        value = value.split(':')[-1].strip()
        return cls(value)

    @classmethod
    def sheet(cls, sheet: IceSheet) -> "RignotBasin":
        sheets = {
            cls.rK_A: IceSheet.wais,
            cls.rA_Ap: IceSheet.wais,
            cls.rAp_B: IceSheet.wais,
            cls.rJpp_K: IceSheet.wais,
            cls.rB_C: IceSheet.wais,
            cls.rC_Cp: IceSheet.wais,
            cls.rCP_D: IceSheet.wais,
            cls.rE_Ep: IceSheet.wais,
            cls.rDp_E: IceSheet.wais,
            cls.rD_Dp: IceSheet.wais,
            cls.rJ_Jpp: IceSheet.eais,
            cls.rEp_f: IceSheet.eais,
            cls.rF_G: IceSheet.eais,
            cls.rG_H: IceSheet.eais,
            cls.rH_Hp: IceSheet.eais,
            cls.rHp_I: IceSheet.apis,
            cls.rI_Ipp: IceSheet.apis,
            cls.rIpp_J: IceSheet.apis,
            cls.rNO: IceSheet.gris,
            cls.rNE: IceSheet.gris,
            cls.rSE: IceSheet.gris,
            cls.rSW: IceSheet.gris,
            cls.rCW: IceSheet.gris,
            cls.rNW: IceSheet.gris
        }

        ais = [IceSheet.apis, IceSheet.wais, IceSheet.eais]
        for basin in cls:
            if sheets[basin] == sheet:
                yield basin
            elif sheets[basin] in ais and sheet == IceSheet.ais:
                yield basin
    # antarctica basins:
    rK_A = "K-A"
    rA_Ap = "A-AP"
    rAp_B = "AP-B"
    rJpp_K = "JPP-K"
    rB_C = "B-C"
    rC_Cp = "C-CP"
    rCP_D = "CP-D"
    rE_Ep = "E-EP"
    rDp_E = "DP-E"
    rD_Dp = "D-DP"
    rJ_Jpp = "J-JPP"
    rEp_f = "EP-F"
    rF_G = "F-G"
    rG_H = "G-H"
    rH_Hp = "H-HP"
    rHp_I = "HP-I"
    rI_Ipp = "I-IPP"
    rIpp_J = "IPP-J"
    # greenland basins:
    rNO = "NO"
    rNE = "NE"
    rSE = "SE"
    rSW = "SW"
    rCW = "CW"
    rNW = "NW"
