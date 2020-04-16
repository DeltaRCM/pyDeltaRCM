from pyDeltaRCM import BmiDelta

if __name__ == '__main__':

    delta = BmiDelta()

    delta.initialize('deltaRCM.yaml')

    delta.update_until(1000)

    delta.finalize()
