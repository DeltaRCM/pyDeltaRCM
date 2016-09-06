from pyDeltaRCM import BmiDelta

if __name__ == '__main__':
    delta = BmiDelta()
    delta.initialize('bmi.yaml')

    for time in range(delta.n_steps):
        delta.update_until(time)
