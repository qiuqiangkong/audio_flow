def write_lines(path, strs: list[str]):
	with open(path, "w", encoding="utf-8") as fw:
		for s in strs:
			fw.write(s + "\n")


def read_lines(path) -> list[str]:
	with open(path, "r", encoding="utf-8") as f:
		return f.read().splitlines()


