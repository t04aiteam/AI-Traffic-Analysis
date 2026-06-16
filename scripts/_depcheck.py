import torchvision
print("torchvision", torchvision.__version__)
try:
    import torchvision.transforms.functional_tensor  # noqa: F401
    print("ok   functional_tensor present")
except Exception as e:
    print(f"MISS functional_tensor: {type(e).__name__}: {e}")
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet  # noqa: F401
    from realesrgan import RealESRGANer  # noqa: F401
    print("ok   basicsr + realesrgan import")
except Exception as e:
    print(f"MISS realesrgan/basicsr: {type(e).__name__}: {e}")
