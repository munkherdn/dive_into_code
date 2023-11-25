import math

THICKNESS = 0.00008 # [m] meters

height_of_Fuji = 3776 # [m] meters
folds = 0
thickness = THICKNESS
while thickness <= height_of_Fuji:
    folds = folds + 1
    thickness = thickness * 2

print("To height of Mt. Fuji: {} [folds]".format(folds))

proxima_centauri = 4.0175e+16
def nfolds(thickness, heigth) :
    folds = 0
    while thickness <= heigth :
        folds = folds + 1
        thickness = thickness * 2
    return folds

folds_to_centauri = nfolds(THICKNESS, proxima_centauri)
print("Folds to Proxima Centauri: ", folds_to_centauri)

def len_paper(tickness, number) :
    n = math.pow(2, number)
    return math.pi*tickness/6*(n+4)*(n-1)

def len_paper2(thickness, height) :
    folds = nfolds(thickness=thickness, heigth=height)
    n = math.pow(2, folds)
    return math.pi*thickness/6*(n+4)*(n-1)

print("Length of paper to Moon: ", len_paper(THICKNESS, 43))
print("Length of paper to Mt. Fuji: ", len_paper(THICKNESS, folds))
print("Length of paper to Proxima Centauri: ", len_paper2(THICKNESS, proxima_centauri))
