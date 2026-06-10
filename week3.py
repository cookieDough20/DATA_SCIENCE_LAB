from scipy.spatial import distance

a=[2,4,6]
b=[5,1,9]

print("Euclidean:",distance.euclidean(a,b))
print("Manhattan:",distance.cityblock(a,b))
print("Minkowski:",distance.minkowski(a,b,p=3))
