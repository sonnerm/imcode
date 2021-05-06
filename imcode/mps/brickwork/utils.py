from tenpy.linalg.charges import LegCharge
from tenpy.networks.site import Site
class BrickworkSite(Site):
    def __init__(self):
        super().__init__(LegCharge.from_trivial(4),["+","b","a","-"],)
