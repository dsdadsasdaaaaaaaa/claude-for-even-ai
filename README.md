# Claude for Even AI

A personal Home Assistant integration that answers Even AI on Even Realities G2 glasses with Claude, through the
Claude API, and can hand jobs to Claude Code on a Windows laptop.

It is a personal project, published here only so Home Assistant can install its updates through HACS. It comes with no
support.

## Installing it with HACS

1. In HACS, open the menu (three dots, top right) and choose **Custom repositories**. Add this repository's address
   with the type **Integration**.
2. Find **Claude for Even AI** in HACS, download it, and restart Home Assistant.
3. Go to **Settings > Devices & services > Add integration** and choose **Claude for Even AI**. It asks for an
   Anthropic API key, then shows the endpoint and key to enter under Agent Configuration in the Even app.

Updates then appear in Home Assistant like any other update.
