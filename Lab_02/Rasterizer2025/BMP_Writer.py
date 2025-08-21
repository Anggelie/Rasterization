import struct

def GenerateBMP(filename: str, width: int, height: int, byteDepth: int, colorBuffer: list[list[tuple[int, int, int]]]) -> None:
    
    def char(c: str) -> bytes:
        # 1 byte
        return struct.pack("<c", c.encode("ascii"))

    def word(w: int) -> bytes:
        # 2 bytes
        return struct.pack("<H", w)

    def dword(d: int) -> bytes:
        # 4 bytes
        return struct.pack("<L", d)

    with open(filename, "wb") as file:
        # Header
        file.write(char("B"))
        file.write(char("M"))
        file.write(dword(14 + 40 + (width * height * byteDepth)))
        file.write(dword(0))
        file.write(dword(14 + 40))
        
        # Info Header
        file.write(dword(40))
        file.write(dword(width))
        file.write(dword(height))
        file.write(word(1))
        file.write(word(byteDepth * 8))
        file.write(dword(0))
        file.write(dword(width * height * byteDepth))
        file.write(dword(0))
        file.write(dword(0))
        file.write(dword(0))
        file.write(dword(0))
        
        # Color table
        for y in range(height):
            for x in range(width):
                color = colorBuffer[x][y]
                for i in range(len(color) - 1, -1, -1):
                    file.write(color[i].to_bytes(1, "little"))
