# Interface Usage Notes

The package includes a dependency-free static visualization viewer.

From the Week 6 project folder on the server:

```bash
cd ~/PigBench/Week6_Unibo_Dataset_Validation
python -m http.server 8506
```

Then open:

```text
http://localhost:8506/interface_demo/week6_static_visualization_viewer.html
```

If using Visual Studio Code Remote, forward port 8506 from the Ports panel.

To stop the server, press Control + C in the terminal.
Do not use Control + Z because it only suspends the process.
