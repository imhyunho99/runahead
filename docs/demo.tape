# runahead 데모 GIF
#
#   pip install runahead      # runahead 가 PATH 에 있어야 한다
#   brew install vhs
#   vhs docs/demo.tape        # → docs/demo.gif
#
# simulate 는 합성 사용자를 상대로 학습 루프만 돌린다.
# 에이전트 CLI 도, API 키도, 토큰 비용도 필요 없다 — 그래서 데모로 쓴다.

Output docs/demo.gif

Set Shell "bash"
Set FontSize 17
Set Width 1180
Set Height 720
Set Padding 22
Set Theme "Catppuccin Mocha"
Set TypingSpeed 38ms
Set PlaybackSpeed 1.0

Hide
Type "clear" Enter
Show

Sleep 800ms

Type "runahead --help"
Sleep 400ms
Enter
Sleep 3500ms

Type "clear"
Enter
Sleep 400ms

Type "# 예측이 비용만큼 값을 하는가? 합성 사용자로 학습 루프를 돌려본다"
Sleep 600ms
Enter
Sleep 400ms

Type "runahead simulate --sessions 500 --seed 42"
Sleep 400ms
Enter
Sleep 5500ms
