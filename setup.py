import os
import setuptools

setuptools.setup(
    name='chainsaw',
    version='1.0',
    keywords='Chainsaw',
    description='Deep learning package',
    long_description=open(
        os.path.join(
            os.path.dirname(__file__),
            'README.md'
        )
    ).read(),
    include_package_data=True,
    package_data={'': ['.txt']},
    author='Ziv Yong',
    author_email='yongxulong@gmail.com',
    url='https://xxx/xxxx/packagedemo',
    packages=setuptools.find_packages(),
    license='MIT'
)

# 以下命令打包
# python setup.py bdist_wheel --universal
# 其中 universal 表示同时支持py2和py3
