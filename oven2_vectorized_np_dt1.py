

"""industrial conveyor oven simulator 
   author: John Morrow

Classes:
--------
OvenClass
    builds and runs an oven instance

""" 

import numpy as np


class OvenClass(object):
    """
    Attributes
    ----------

    _Cp: (float)
        specific heat of product material
    _D: (float)
        density of product material
    _K: (float)
        thermal conductivity of product material in-plane
    _H: (float)
        heat transfer coefficient of oven gas
    _pl: (float)
        product length (along axis of belt movement)
    _pw: (float)
        product width
    _pt: (float)
        product thickness
    _Tamb: (float)
        ambient temperature
    _LH: (numpy array)
        heater lengths
    _bottom: flag is True when lower heaters are also active

    Methods
    -------
    run_oven

    """

    def __init__(self, Cp=1.3e3, H=50.0, D=2.0e3, K=0.8, pw=0.1, pl=0.1,
                 pt=0.0016, Tamb=20.0, LH=np.array([0.3048, 0.3048, 0.3048, 0.3048, 0.6]),
                 bottom=False):

        """
        Parameters
        ----------
        Cp: (float)
            specific heat of product material, J/kg-degC
            default: 1.3e3 
            example: FR4 substrate = 1.3e3
        D: (float)
            density of product material, kg/m**3
            default: 2.0e3 
            example: FR4 substrate = 2.0e3 
        K: (float)
            thermal conductivity of product material in-plane, w/m-degK
            default: 0.8
            example: FR4 in-plane = 0.8 
        H: (float)
            heat transfer coefficient of oven gas, W/degK-m**2
            default: 50.0 
            example: air = 50
        pl: (float)
            product length (along axis of belt movement), meters
            default: 0.1 
        pw: (float)
            product width, meters
            default: 0.1 
        pt: (float)
            product thickness, meters
            default: 0.0016 [0.062"
        Tamb: (float)
            ambient temperature, i.e. initial product temperature, degC
            default: 20.0  
        LH: (numpy array)
            heater lengths , meters
            example: ([0.3048,0.3048,0.3048,0.3048,0.6])
        bottom: flag is True when lower heaters are also active
        
        """

        self._Cp = Cp
        self._H = H 
        self._D = D 
        self._K = K 
        self._pw = pw
        self._pl = pl
        self._pt = pt
        self._Tamb = Tamb 
        self._LH = LH
        self._bottom = bottom
        
        if self._bottom is True:
            self._bflag = 2.0
        else:
            self._bflag = 1.0

    def run_oven(self, Thtr, bv):
        """Run one pass through oven.
        
        Parameters
        ----------
        Thtr: (numpy array)
            heating elements temperatures, degC
            example: ([230,150,270,220,20])
            note: length of Thtr must equal length of LH
        bv: (float)
            belt speed, meters/sec
        
        Returns
        -------
        (numpy array) temperature at center of product at 1 second intervals (for dt -0.1)
         
        """

        dt = 1.0  # seconds, delta time of a segment --heater & product segments
        A_thru = self._pl * self._pw  # through-plane area of one side of product
        self._mass = self._D * A_thru * self._pt  # total mass of product
        num_prod_segs = round(self._pl / (bv * dt))  # was int()
        k2 = self._H * (A_thru * self._bflag / num_prod_segs) * dt  # convection
        k1 = self._K * self._pw * self._pt * dt / (self._pl/num_prod_segs)  # conduction
        k3 = self._Cp * (self._mass / num_prod_segs)  # Q -> T
        
        if self._LH.size != Thtr.size:
            print("error: number settings (Thtr) not equal elements (LH)")
        if k1/k3 > 0.5:
            print(f"warning: simulator unstable, k3/k1 ={(k3/k1):.2f}, must be <= 0.5")
        
        # 'elements' holds the temperatures of heater segments
        elements = np.array([])
        for i in range(len(self._LH)):
            htr_segs = np.zeros(round(self._LH[i]/(bv * dt)))  # was round()  sets up 1/dt segments per second
            htr_segs.fill(Thtr[i])
            elements = np.concatenate([elements, htr_segs])

        # 'product' holds the temperatures of product segments
        product = np.full(num_prod_segs, self._Tamb)
        ctr_seg_idx = int(num_prod_segs/2)
        deltaQr = np.zeros(num_prod_segs)
        deltaQl = np.zeros(num_prod_segs)
        elements_plus = np.concatenate([product, elements, product])
        steps = len(elements) + num_prod_segs
        Temp = np.zeros(steps)  # final product (centers) profile

        product_plus = np.array([])
        for j in range(0, steps):
            # 'product_plus' augments 'product' for tf.roll operation
            product_plus = np.pad(product, (1, 1), 'edge')
            # product_plus = np.append(np.insert(product, 0, product[0]), product[num_prod_segs - 1])
            product_left = np.roll(product_plus, shift=1, axis=0)[1:num_prod_segs + 1]
            product_right = np.roll(product_plus, shift=-1, axis=0)[1:num_prod_segs + 1]
            q_sum = (product_left + product_right - 2.0 * product) * k1
            q_sum2 = (elements_plus[j:j + num_prod_segs] - product) * k2
            product = (q_sum + q_sum2) / k3 + product
            Temp[j] = product[ctr_seg_idx]

        
        return Temp[ctr_seg_idx:steps:int(1 / dt)]

