
import hashlib

def sha256_iterate(start, times):
	value = str(start).encode()
	for i in range(times):
		value = hashlib.sha256(value).digest()
		print(f"第{i+1}次迭代: {value.hex()}")

if __name__ == "__main__":
	sha256_iterate(1, 10)  # 迭代10次，可根据需要修改
