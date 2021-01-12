from tenpy.linalg.charges import LegCharge
from tenpy.networks.site import Site
class FlatSite(Site):
    def __init__(self):
        super().__init__(LegCharge.from_trivial(2),["d","u"],)
def flat_to_folded(mps,chi):
    pass
