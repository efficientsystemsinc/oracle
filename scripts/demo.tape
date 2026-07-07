# vhs scripts/demo.tape  ->  assets/demo.gif
Output assets/demo.gif

Set Shell "bash"
Set FontSize 15
Set Width 1180
Set Height 640
Set Padding 16
Set Theme { "background": "#ffffff", "foreground": "#1a1a1a", "cursor": "#1a1a1a", "black": "#1a1a1a", "red": "#8a3033", "green": "#3a6b35", "yellow": "#8a6d1a", "blue": "#2a4d69", "magenta": "#6b3a62", "cyan": "#2a6969", "white": "#666666", "brightBlack": "#666666", "brightRed": "#8a3033", "brightGreen": "#3a6b35", "brightYellow": "#8a6d1a", "brightBlue": "#2a4d69", "brightMagenta": "#6b3a62", "brightCyan": "#2a6969", "brightWhite": "#1a1a1a" }
Set TypingSpeed 40ms

Type "oracle query 'pgbouncer asyncpg setting'"
Sleep 500ms
Enter
Sleep 5s

Type "clear"
Enter
Type "oracle query --as-of 2026-06-20 'perseus prod database'   # time travel"
Sleep 500ms
Enter
Sleep 5s

Type "clear"
Enter
Type "oracle ask 'why does asyncpg need statement_cache_size=0 in prod'"
Sleep 500ms
Enter
Hide
Sleep 20s
Show
Sleep 7s

Type "clear"
Enter
Type "curl -s localhost:4141/health | jq .latency_ms"
Sleep 500ms
Enter
Sleep 4s
