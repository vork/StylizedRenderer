import platform
import numpy as np
import pytest
from inkforge.procedural import showroom
from inkforge.renderer import Renderer,Style


@pytest.mark.skipif(platform.system()!='Darwin',reason='Metal fallback path')
def test_storage_geometry_matches_five_pass_materials_and_occlusion():
    r=Renderer(showroom(),320,200,shadow_size=256);s=Style(taa=False)
    r.storage_geometry=False;r.render([4,4,-7],[0,.5,1],s)
    expected=[t.to_numpy().copy() for t in r.gbuffers]
    before=r.output.to_numpy().copy()
    r.storage_geometry=True;r.render([4,4,-7],[0,.5,1],s)
    for reference,target in zip(expected,r.gbuffers):
        np.testing.assert_allclose(target.to_numpy(),reference,atol=2e-3,rtol=1e-4)
    assert abs(r.output.to_numpy()-before).mean()<.001
