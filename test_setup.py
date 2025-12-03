import earthquake as libs

print('✓ earthquake module imported successfully')
print('\nAvailable packages:')
availability = libs.availability()
for key, value in availability.items():
    print(f'  {key}: {value}')
