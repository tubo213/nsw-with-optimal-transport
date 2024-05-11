# bin/配下にあるスクリプトを全て実行する
# このスクリプトを実行することで、全てのスクリプトを一括で実行できる

for script in $(ls bin); do
  if [ -f bin/$script ]; then
    echo "Running $script"
    bash bin/$script
  fi
done