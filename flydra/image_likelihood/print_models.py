import flydra.image_likelihood.models as models
import sympy

if 1:
    model = models.TargetModel()
    x = models.SymbolicStateVector(sympy.DeferredVector('x'))
    Fsym = model.get_process_model_ODEs_linearized(x)
    sympy.pprint(Fsym)
