class PostingCompressor:

    @staticmethod
    def vb_encode_number(n: int) -> list[int]:
        bytes_ = []
        while True:
            bytes_.insert(0, n % 128)
            if n < 128:
                break
            n //= 128
        bytes_[-1] += 128
        return bytes_

    @classmethod
    def vb_encode_list(cls, numbers: list[int]) -> bytes:
        if not numbers:
            return b""
        out = []
        for n in numbers:
            out.extend(cls.vb_encode_number(n))
        return bytes(out)

    @staticmethod
    def vb_decode(byte_stream: bytes) -> list[int]:
        numbers, n = [], 0
        for b in byte_stream:
            if b < 128:
                n = 128 * n + b
            else:
                n = 128 * n + (b - 128)
                numbers.append(n)
                n = 0
        return numbers

    @staticmethod
    def gap_encode(numbers: list[int]) -> list[int]:
        if not numbers:
            return []
        gaps = [numbers[0]]
        for i in range(1, len(numbers)):
            gaps.append(numbers[i] - numbers[i - 1])
        return gaps

    @staticmethod
    def gap_decode(gaps: list[int]) -> list[int]:
        if not gaps:
            return []
        numbers = [gaps[0]]
        for i in range(1, len(gaps)):
            numbers.append(numbers[-1] + gaps[i])
        return numbers

    @classmethod
    def compress(cls, numbers: list[int]) -> bytes:
        return cls.vb_encode_list(cls.gap_encode(numbers))

    @classmethod
    def decompress(cls, encoded: bytes) -> list[int]:
        return cls.gap_decode(cls.vb_decode(encoded))
