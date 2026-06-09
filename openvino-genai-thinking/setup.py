from setuptools import setup, find_packages

setup(
    packages=find_packages(),
    include_package_data=True,
    # 标记为平台相关的 wheel（包含 .so）
    has_ext_modules=lambda: True,
)
