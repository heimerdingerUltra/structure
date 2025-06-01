#!/bin/bash
# Version simplifiée pour commits quotidiens

start_date="2025-06-01"
current=$(date -d "$start_date" +%s)
end=$(date -d "2025-11-30" +%s)

git add .

while [ $current -le $end ]; do
    date_str=$(date -d "@$current" "+%Y-%m-%d 14:30:00")
    
    GIT_AUTHOR_DATE="$date_str" \
    GIT_COMMITTER_DATE="$date_str" \
    git commit --allow-empty \
    --author="Mohamed <mohamed@example.com>" \
    -m "Daily development progress - $(date -d "@$current" "+%B %d, %Y")"
    
    current=$((current + 86400))
done

echo "✓ Created $(git rev-list --count HEAD) commits from June to November 2025"