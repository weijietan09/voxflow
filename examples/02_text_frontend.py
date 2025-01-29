"""示例 02：文本前端。

展示中英混排文本如何被归一化、切段并转成音素 / 符号 id 序列。
"""

from voxflow.text.frontend import TextFrontend


def main() -> None:
    frontend = TextFrontend()
    for text in ["你好，世界！", "今天气温23度", "语音克隆 voice cloning"]:
        out = frontend.encode(text)
        print(f"原文: {text}")
        print(f"  归一化: {out.text}")
        print(f"  token: {out.tokens}")
        print(f"  ids  : {out.ids}")
        print()


if __name__ == "__main__":
    main()
