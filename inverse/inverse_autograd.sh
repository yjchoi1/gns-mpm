
for i in {0..20..1}
do
  python3 inverse_autograd.py --epoch=${i} --phi=45.0
done

