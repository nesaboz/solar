import sys
print(sys.path)
from code.processing import BigImage


def test_image_size():
    bi = BigImage('R1C1')
    assert bi.size == (18432, 18432)
