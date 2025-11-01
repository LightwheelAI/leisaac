# Trouble Shooting

## Vscode debugger does not work

> related issue: [IsaacLab/issues/3305](https://github.com/isaac-sim/IsaacLab/issues/3305)

When you launch the program using VSCode Python Debugger, you may encounter the following error:

```shell
OSError: libstdc++.so.6: version `GLIBCXX_3.4.30' not found
```

Please try installing the corresponding dependencies in your conda environment:
```shell
conda install -c conda-forge gcc=12 -y
```
