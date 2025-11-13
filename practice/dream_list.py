import json

print("==== WELCOME TO THE DREAM LIST NOTE BOOK ====")
print("1. Add New Dream")
print("2. Watch The Dream List")
print("3. Remove Any Dream Frome You List")
print("4. exit \n")



while True:
    try:
        number_selection = int(input("\n📋 Select a number between (1-4): "))

        if number_selection == 1:
            itss_true = True
            while itss_true:
                user_input = input("\nWhat's your Dream❓(write your dream or enter exit): ")

                if user_input.lower() == 'exit':
                    itss_true = False
                else:    
                    try:
                        with open("Dream.json", "r") as dream_file:
                            dream_list = json.load(dream_file)
                    except (FileNotFoundError, json.JSONDecodeError):
                        dream_list = {}

                    dream_no = f"Dream_no_{len(dream_list) + 1}"
                    dream_list[dream_no] = user_input

                    with open("Dream.json", "w") as dream_file:
                        json.dump(dream_list, dream_file, indent=4)

                    print(f"\n✅ Successfully added your Dream in list: {user_input}\n")


        elif number_selection == 2:
            try:
                with open("Dream.json","r") as dream:
                    dreams = dict(json.load(dream))
                    if not dreams :
                        print("\n😴 You don't have any dreams saved yet!")
                        
                    else:
                        for key, value in dreams.items():
                            print(f"✨ {key}: {value}\n")
                        print("=" * 50, "\n")

            except (FileNotFoundError):
                print(f"\n❌ there was no file. please first make a Dream list❗\n")


        elif number_selection == 3:
            try:
                with open("Dream.json", "r") as dreamss:
                    dream_list = dict(json.load(dreamss))
           
                    if not dream_list:
                        print("\n😴 You don't have any dreams saved yet!")
                        continue  
                       
                    print("\n🌙 Your Dream List:")
                    print("=" * 50)

                    for key, value in dream_list.items():
                        print(f"✨ {key}: {value}\n")
                        
                    print("=" * 50, "\n")
           
                    remove_key = input("\n📝 Enter the dream number you want to remove: ")

                    if ',' in remove_key:
                        removed = remove_key.strip().split(',')
                        for r in removed:
                            dream_key = f"Dream_no_{int(r)}"

                            if dream_key in dream_list:
                                del dream_list[dream_key]
                                print(f"\n✅ Removed Dream {r} successfully!")
           
                            else:
                                print("⚠️ Dream not found!")
                       
                            with open("Dream.json", "w") as files:
                                json.dump(dream_list, files, indent=4)
            
                    else:
                        removed = remove_key.strip().split(' ')
                        for r in removed:
                            dream_key = f"Dream_no_{int(r)}"

                            if dream_key in dream_list:
                                del dream_list[dream_key]
                                print(f"\n✅ Removed Dream {r} successfully!")
           
                            else:
                                print("⚠️ Dream not found!")
                       
                            with open("Dream.json", "w") as files:
                                json.dump(dream_list, files, indent=4)
            

            except FileNotFoundError:
                print("\n❌ You don't have a dream list yet! Please add one first.")



        elif number_selection == 4:
            print("\n👋 Exiting program... Goodbye!\n")
            break

    except ValueError:
        print("\n❌ Please Enter Valid Numbers❗\n")
        continue
