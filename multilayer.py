class Substrate:
    """XRR 기판을 나타내는 클래스 (3개 파라미터)."""

    def __init__(self, sld: float, roughness: float, density: float):
        self.sld = sld
        self.roughness = roughness
        self.density = density

    def __repr__(self) -> str:
        return f"Substrate(sld={self.sld:.4f}, roughness={self.roughness:.4f}, density={self.density:.4f})"


class Layer:
    """XRR 단일 층을 나타내는 클래스 (4개 파라미터)."""

    def __init__(self, thickness: float, sld: float, roughness: float, density: float):
        self.thickness = thickness  # layer thickness
        self.sld = sld  # disp / n*b substrate
        self.roughness = roughness  # sigma layer in A
        self.density = density  # di_nb/beta layer

    def __repr__(self) -> str:
        return (
            f"Layer(thickness={self.thickness:.4f}, sld={self.sld:.4f}, "
            f"roughness={self.roughness:.4f}, density={self.density:.4f})"
        )


class MultiLayer:
    """XRR 다층 구조 전체를 나타내는 클래스."""

    def __init__(self, substrate: Substrate, layers: list[Layer]):
        self.substrate = substrate
        self.layers = layers

    def __repr__(self) -> str:
        layers_repr = "\n  ".join(str(layer) for layer in self.layers)
        return f"MultiLayer(\n  {self.substrate},\n  {layers_repr}\n)"

    def get_total_thickness(self) -> float:
        """전체 층의 총 두께를 반환 (기판 제외)."""
        return sum(layer.thickness for layer in self.layers)
