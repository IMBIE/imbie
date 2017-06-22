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
            cls.rK_A: IceSheet.eais,
            cls.rA_Ap: IceSheet.eais,
            cls.rAp_B: IceSheet.eais,
            cls.rJpp_K: IceSheet.eais,
            cls.rB_C: IceSheet.eais,
            cls.rC_Cp: IceSheet.eais,
            cls.rCP_D: IceSheet.eais,
            cls.rE_Ep: IceSheet.eais,
            cls.rDp_E: IceSheet.eais,
            cls.rD_Dp: IceSheet.eais,
            cls.rJ_Jpp: IceSheet.wais,
            cls.rEp_f: IceSheet.wais,
            cls.rF_G: IceSheet.wais,
            cls.rG_H: IceSheet.wais,
            cls.rH_Hp: IceSheet.wais,
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
