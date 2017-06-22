from .sheets import IceSheet
import enum


class ZwallyBasin(enum.Enum):
    @classmethod
    def parse(cls, value: str) -> "ZwallyBasin":
        value = value.split(':')[-1].strip()
        if len(value) < 2:
            value = '0' + value
        return cls(value)

    @classmethod
    def sheet(cls, sheet: IceSheet) -> "ZwallyBasin":
        sheets = {
            cls.z01: IceSheet.wais,
            cls.z02: IceSheet.eais,
            cls.z03: IceSheet.eais,
            cls.z04: IceSheet.eais,
            cls.z05: IceSheet.eais,
            cls.z06: IceSheet.eais,
            cls.z07: IceSheet.eais,
            cls.z08: IceSheet.eais,
            cls.z09: IceSheet.eais,
            cls.z10: IceSheet.eais,
            cls.z11: IceSheet.eais,
            cls.z12: IceSheet.eais,
            cls.z13: IceSheet.eais,
            cls.z14: IceSheet.eais,
            cls.z15: IceSheet.eais,
            cls.z16: IceSheet.eais,
            cls.z17: IceSheet.eais,
            cls.z18: IceSheet.wais,
            cls.z19: IceSheet.wais,
            cls.z20: IceSheet.wais,
            cls.z21: IceSheet.wais,
            cls.z22: IceSheet.wais,
            cls.z23: IceSheet.wais,
            cls.z24: IceSheet.apis,
            cls.z25: IceSheet.apis,
            cls.z26: IceSheet.apis,
            cls.z27: IceSheet.apis,
            cls.z1_1: IceSheet.gris,
            cls.z1_2: IceSheet.gris,
            cls.z1_3: IceSheet.gris,
            cls.z1_4: IceSheet.gris,
            cls.z2_1: IceSheet.gris,
            cls.z2_2: IceSheet.gris,
            cls.z3_1: IceSheet.gris,
            cls.z3_2: IceSheet.gris,
            cls.z3_3: IceSheet.gris,
            cls.z4_1: IceSheet.gris,
            cls.z4_2: IceSheet.gris,
            cls.z4_3: IceSheet.gris,
            cls.z5_0: IceSheet.gris,
            cls.z6_1: IceSheet.gris,
            cls.z6_2: IceSheet.gris,
            cls.z7_1: IceSheet.gris,
            cls.z7_2: IceSheet.gris,
            cls.z8_1: IceSheet.gris,
            cls.z8_2: IceSheet.gris
        }
        ais = [IceSheet.apis, IceSheet.wais, IceSheet.eais]
        for basin in cls:
            if sheets[basin] == sheet:
                yield basin
            elif sheets[basin] in ais and sheet == IceSheet.ais:
                yield basin

    # antarctica basins:
    z01 = "01"
    z02 = "02"
    z03 = "03"
    z04 = "04"
    z05 = "05"
    z06 = "06"
    z07 = "07"
    z08 = "08"
    z09 = "09"
    z10 = "10"
    z11 = "11"
    z12 = "12"
    z13 = "13"
    z14 = "14"
    z15 = "15"
    z16 = "16"
    z17 = "17"
    z18 = "18"
    z19 = "19"
    z20 = "20"
    z21 = "21"
    z22 = "22"
    z23 = "23"
    z24 = "24"
    z25 = "25"
    z26 = "26"
    z27 = "27"
    # greenland basins
    z1_1 = "1.1"
    z1_2 = "1.2"
    z1_3 = "1.3"
    z1_4 = "1.4"
    z2_1 = "2.1"
    z2_2 = "2.2"
    z3_1 = "3.1"
    z3_2 = "3.2"
    z3_3 = "3.3"
    z4_1 = "4.1"
    z4_2 = "4.2"
    z4_3 = "4.3"
    z5_0 = "5.0"
    z6_1 = "6.1"
    z6_2 = "6.2"
    z7_1 = "7.1"
    z7_2 = "7.2"
    z8_1 = "8.1"
    z8_2 = "8.2"
